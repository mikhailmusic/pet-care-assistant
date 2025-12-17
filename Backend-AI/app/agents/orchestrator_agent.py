# app/agents/orchestrator_agent.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime, timezone
import asyncio
from loguru import logger
import json
import operator

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from app.integrations.gigachat_client import create_llm_from_settings
from app.dto import ChatSettingsDTO
from app.models.message import Message


# ============================================================================
# STATE
# ============================================================================

class AgentState(TypedDict):
    """
    Общий State для всей мультиагентной системы.

    Базируется на MessagesState (messages: List[BaseMessage]) +
    кастомные поля для контекста и результатов.
    """
    # Messages (из MessagesState)
    messages: Annotated[List[BaseMessage], operator.add]

    # Основной контекст
    user_id: int
    chat_id: int
    uploaded_files: List[Dict[str, Any]]
    chat_settings: Dict[str, Any]

    # Контекст питомцев (для агентов)
    current_pet_id: Optional[int]
    current_pet_name: str
    known_pets: List[Dict[str, Any]]

    # Результаты работы агентов (аккумулируются)
    agent_results: Annotated[List[Dict[str, Any]], operator.add]
    generated_files: Annotated[List[Dict[str, Any]], operator.add]

    # Routing (куда идти дальше)
    next_agent: Optional[str]

    # Итоговый ответ
    final_response: Optional[str]


# ============================================================================
# RESULT
# ============================================================================

@dataclass
class OrchestratorResult:
    """Результат работы оркестратора"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_files: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# ORCHESTRATOR AGENT (LangGraph)
# ============================================================================

class OrchestratorAgent:
    """
    Главный оркестратор - координирует работу всех специализированных агентов.

    Использует LangGraph StateGraph с Supervisor Pattern:
    - START → supervisor
    - supervisor → агенты (conditional routing)
    - агенты → supervisor (return edges)
    - supervisor → END (когда готов финальный ответ)

    Агенты:
    1. pet_memory - работа с БД питомцев и здоровья
    2. document_rag - поиск в документах и индексация
    3. multimodal - анализ изображений, видео, аудио
    4. web_search - поиск в интернете
    5. health_nutrition - анализ здоровья и питания
    6. calendar - работа с Google Calendar
    7. content_generation - генерация контента
    8. email - отправка писем
    """

    def __init__(
        self,
        pet_memory_agent,
        document_rag_agent,
        multimodal_agent,
        web_search_agent,
        health_nutrition_agent,
        calendar_agent,
        content_generation_agent,
        email_agent,
        llm=None,
        llm_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
        max_iterations: int = 5,
    ):
        self.pet_memory_agent = pet_memory_agent
        self.document_rag_agent = document_rag_agent
        self.multimodal_agent = multimodal_agent
        self.web_search_agent = web_search_agent
        self.health_nutrition_agent = health_nutrition_agent
        self.calendar_agent = calendar_agent
        self.content_generation_agent = content_generation_agent
        self.email_agent = email_agent

        # LLM factory builds client using chat settings from DB (fallback to env defaults)
        self._llm_factory = llm_factory or create_llm_from_settings
        self._base_llm = llm or self._llm_factory({})
        self.llm = self._base_llm

        # Prevent concurrent runs from clobbering per-chat LLM binding.
        self._lock = asyncio.Lock()
        self.max_iterations = max_iterations

        # Строим граф
        self.graph = self._build_graph()

        logger.info("OrchestratorAgent (LangGraph) initialized with 8 specialized agents")

    def _build_graph(self) -> StateGraph:
        """Построить LangGraph для мультиагентной системы"""

        workflow = StateGraph(AgentState)

        # Добавляем узлы
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("pet_memory", self._create_agent_node("pet_memory", self.pet_memory_agent))
        workflow.add_node("document_rag", self._create_agent_node("document_rag", self.document_rag_agent))
        workflow.add_node("multimodal", self._create_agent_node("multimodal", self.multimodal_agent))
        workflow.add_node("web_search", self._create_agent_node("web_search", self.web_search_agent))
        workflow.add_node("health_nutrition", self._create_agent_node("health_nutrition", self.health_nutrition_agent))
        workflow.add_node("calendar", self._create_agent_node("calendar", self.calendar_agent))
        workflow.add_node("content_generation", self._create_agent_node("content_generation", self.content_generation_agent))
        workflow.add_node("email", self._create_agent_node("email", self.email_agent))

        # START → supervisor
        workflow.add_edge(START, "supervisor")

        # Return edges: агенты → supervisor
        workflow.add_edge("pet_memory", "supervisor")
        workflow.add_edge("document_rag", "supervisor")
        workflow.add_edge("multimodal", "supervisor")
        workflow.add_edge("web_search", "supervisor")
        workflow.add_edge("health_nutrition", "supervisor")
        workflow.add_edge("calendar", "supervisor")
        workflow.add_edge("content_generation", "supervisor")
        workflow.add_edge("email", "supervisor")

        # Conditional edges: supervisor → агенты или END
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state.get("next_agent", END),
            {
                "pet_memory": "pet_memory",
                "document_rag": "document_rag",
                "multimodal": "multimodal",
                "web_search": "web_search",
                "health_nutrition": "health_nutrition",
                "calendar": "calendar",
                "content_generation": "content_generation",
                "email": "email",
                END: END,
            }
        )

        return workflow.compile()

    def _bind_llm(self, chat_settings: Dict[str, Any]):
        """
        Возвращает LLM, привязанную к настройкам чата.
        Используем фабрику, которая создает клиент по данным из БД.
        """
        settings = self._normalize_chat_settings(chat_settings)
        if not settings:
            return self._base_llm

        try:
            bound_llm = self._llm_factory(settings) or self._base_llm
        except Exception as e:
            logger.error(f"LLM factory failed, falling back to base LLM: {e}")
            bound_llm = self._base_llm

        return bound_llm

    @staticmethod
    def _normalize_chat_settings(chat_settings: Dict[str, Any] | None) -> Dict[str, Any]:
        if chat_settings is None:
            return {}
        if hasattr(chat_settings, "model_dump"):
            try:
                return chat_settings.model_dump()
            except Exception:
                return dict(chat_settings)
        settings = dict(chat_settings)
        if settings.get("temperature") is not None:
            try:
                temp = float(settings["temperature"])
                settings["temperature"] = max(0.0, min(1.0, temp))
            except Exception:
                settings.pop("temperature", None)
        return settings

    @staticmethod
    def _parse_decision(decision_text: str) -> Dict[str, Any]:
        """
        Parse supervisor JSON with resilience to trailing commas or minor formatting issues.
        """
        import re

        json_match = re.search(r'\{.*\}', decision_text, re.DOTALL)
        if not json_match:
            return {"action": "finish", "reason": "Не удалось распарсить решение"}

        raw = json_match.group(0)
        try:
            return json.loads(raw)
        except Exception:
            cleaned = re.sub(r',(\s*[}\]])', r'\1', raw)
            try:
                return json.loads(cleaned)
            except Exception as e:
                logger.error(f"Failed to parse supervisor decision: {e}")
                return {"action": "finish", "reason": f"Ошибка парсинга: {str(e)}"}

    def _supervisor_node(self, state: AgentState) -> AgentState:
        """
        Supervisor узел - анализирует ситуацию и решает что делать дальше.

        Логика:
        1. Смотрит на последнее сообщение пользователя
        2. Смотрит на результаты предыдущих агентов (если есть)
        3. Решает: нужен ли ещё агент или уже можно формировать ответ
        4. Если нужен агент → устанавливает next_agent
        5. Если готов ответ → устанавливает next_agent=END и формирует final_response
        """

        logger.info(f"Supervisor: analyzing state (user={state['user_id']}, chat={state['chat_id']})")

        # Получаем последнее сообщение пользователя
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        last_user_message = user_messages[-1].content if user_messages else ""

        # Уже вызванные агенты
        called_agents = [r["agent"] for r in state.get("agent_results", [])]

        # Предохранитель от бесконечных циклов
        if len(called_agents) >= self.max_iterations:
            logger.warning(f"Max iterations reached ({self.max_iterations}), finishing")
            state["final_response"] = self._build_final_response(state, fallback_to_llm=True)
            state["next_agent"] = END
            return state

        # Настройки чата
        settings = state["chat_settings"]
        uploaded_files = state.get("uploaded_files", [])
        

        # Формируем промпт для supervisor LLM
        system_prompt = self._build_supervisor_prompt(
            settings=settings,
            uploaded_files=uploaded_files,
            called_agents=called_agents,
        )

        # Собираем context для LLM
        context_messages = [SystemMessage(content=system_prompt)]

        # Добавляем результаты агентов если есть
        for result in state.get("agent_results", []):
            agent_name = result["agent"]
            agent_output = result["output"]
            context_messages.append(
                AIMessage(content=f"[Результат {agent_name}]\n{agent_output}")
            )

        # Последнее сообщение пользователя
        context_messages.append(HumanMessage(content=last_user_message))

        # Формируем инструкцию с учётом настроек
        enabled_features = []
        disabled_features = []

        if settings.get("web_search_enabled"):
            enabled_features.append("веб-поиск (используй активно для актуальной информации)")
        else:
            disabled_features.append("web_search")

        if settings.get("image_generation_enabled"):
            enabled_features.append("генерация изображений")
        else:
            disabled_features.append("content_generation (генерация изображений)")

        if settings.get("voice_response_enabled"):
            enabled_features.append("голосовой ответ (если пользователь просит аудио/голос)")

        enabled_text = f"\n✅ Включено: {', '.join(enabled_features)}" if enabled_features else ""
        disabled_text = f"\n❌ Отключено: {', '.join(disabled_features)}" if disabled_features else ""

        # Просим LLM принять решение
        decision_prompt = f"""Проанализируй ситуацию и реши что делать дальше.

Верни JSON:
{{
  "action": "call_agent" | "finish",
  "agent": "pet_memory" | "document_rag" | "multimodal" | "web_search" | "health_nutrition" | "calendar" | "content_generation" | "email" | null,
  "reason": "почему это решение",
  "final_response": "итоговый ответ пользователю" (если action=finish)
}}

**НАСТРОЙКИ ЧАТА:**{enabled_text}{disabled_text}

**ПРАВИЛА:**
1. **Веб-поиск (web_search):**
   - Если web_search_enabled=True → ИСПОЛЬЗУЙ web_search для любой информации, требующей актуальных данных
   - Если web_search_enabled=False → НЕ вызывай web_search, объясни что функция отключена

2. **Генерация контента (content_generation):**
   - Если image_generation_enabled=True → можешь использовать для генерации изображений/графиков
   - Если image_generation_enabled=False → НЕ вызывай content_generation для генерации, объясни что отключено

3. **Голосовой ответ (КРИТИЧЕСКИ ВАЖНО!):**
   - Если voice_response_enabled=True И пользователь ЯВНО просит "в аудио", "голосом", "озвучь":
     * Шаг 1: Получи информацию (вызови нужного агента: pet_memory, health_nutrition, и т.д.)
     * Шаг 2: Вызови content_generation для преобразования ответа в аудио
     * НЕ ЗАКАНЧИВАЙ текстовым ответом, если пользователь просил аудио!
   - Если voice_response_enabled=False И пользователь просит аудио → объясни что голосовой ответ отключен

4. **Общие правила:**
   - Ты не просто роутер: если данных недостаточно, задай уточняющий вопрос в final_response и заверши (action=finish)
   - Не вызывай агента повторно, если он уже был вызван и результата достаточно
   - Если можно ответить сразу — finish и дай дружелюбный, полезный ответ
   - ЦЕПОЧКИ АГЕНТОВ: если нужно сначала получить данные, а потом их обработать → вызывай агентов последовательно

Отвечай ТОЛЬКО JSON, без пояснений."""


        context_messages.append(HumanMessage(content=decision_prompt))

        # Вызываем LLM
        llm = self._bind_llm(state.get("chat_settings"))
        try:
            response = llm.invoke(context_messages)
            decision_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Supervisor LLM error: {e}")
            state["final_response"] = (
                "Модель сейчас недоступна. Попробуйте ещё раз позже или выберите другую модель в настройках."
            )
            state["next_agent"] = END
            return state

        # Парсим JSON
        decision = self._parse_decision(decision_text)

        logger.info(f"Supervisor decision: {decision}")


        settings = state.get("chat_settings") or {}
        logger.info(f"Chat settings in supervisor: {settings}")

        agent = decision.get("agent") if decision.get("action") == "call_agent" else None

        # Проверка разрешённых функций
        if agent == "web_search" and not settings.get("web_search_enabled", False):
            state["final_response"] = (
                "В этом чате отключён веб-поиск. "
                "Могу ответить без интернета, либо включи веб-поиск в настройках и повтори запрос."
            )
            state["next_agent"] = END
            return state

        # Проверяем что content_generation разрешён (хотя бы одна из функций)
        if agent == "content_generation":
            if not settings.get("image_generation_enabled", False) and not settings.get("voice_response_enabled", False):
                state["final_response"] = (
                    "В этом чате отключена генерация контента и голосовой ответ. "
                    "Включи нужные функции в настройках — и я смогу создавать изображения, графики, аудио и отчёты."
                )
                state["next_agent"] = END
                return state

        # Применяем решение
        if decision.get("action") == "call_agent":
            next_agent = decision.get("agent")
            if next_agent in called_agents:
                logger.warning(f"Agent {next_agent} already called, finishing to avoid loop")
                state["final_response"] = self._build_final_response(state, fallback_to_llm=True)
                state["next_agent"] = END
                return state
            state["next_agent"] = next_agent
            logger.info(f"Supervisor → routing to: {next_agent}")
        else:
            # Формируем финальный ответ
            if decision.get("final_response"):
                state["final_response"] = decision["final_response"]
            else:
                # Если LLM не вернул ответ, формируем сами
                state["final_response"] = self._build_final_response(state, fallback_to_llm=True)

            state["next_agent"] = END
            logger.info(f"Supervisor → END (response ready)")

        return state

    def _create_agent_node(self, agent_name: str, agent):
        """Создать узел для агента"""

        async def agent_node(state: AgentState) -> AgentState:
            logger.info(f"Agent node: {agent_name} started")

            try:
                # Получаем последнее сообщение пользователя
                user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
                last_user_message = user_messages[-1].content if user_messages else ""

                # Обогащаем сообщение для content_generation если нужен TTS
                agent_message = last_user_message
                agent_results = state.get("agent_results", [])

                if agent_name == "content_generation" and agent_results:
                    # Если есть результаты предыдущих агентов, формируем инструкцию для TTS
                    settings = state.get("chat_settings", {})
                    if settings.get("voice_response_enabled"):
                        # Собираем результаты предыдущих агентов
                        previous_outputs = []
                        for res in agent_results:
                            if not res.get("error"):
                                output = res["output"]
                                # Извлекаем текст из JSON если это JSON
                                try:
                                    if isinstance(output, str) and output.startswith("{"):
                                        data = json.loads(output)
                                        if "analysis" in data:
                                            previous_outputs.append(data["analysis"])
                                        elif "text" in data:
                                            previous_outputs.append(data["text"])
                                        else:
                                            previous_outputs.append(output)
                                    else:
                                        previous_outputs.append(output)
                                except:
                                    previous_outputs.append(output)

                        if previous_outputs:
                            combined_text = "\n\n".join(previous_outputs)
                            agent_message = f"""Озвучь следующий текст через text_to_speech:

{combined_text}

Используй подходящий голос (например, Nec_24000 или Bys_24000) и формат wav16."""

                # Формируем контекст для агента
                context = {
                    "chat_id": state["chat_id"],
                    "uploaded_files": state.get("uploaded_files", []),
                    "chat_settings": state["chat_settings"],
                    "current_pet_id": state.get("current_pet_id"),
                    "current_pet_name": state.get("current_pet_name", ""),
                    "known_pets": state.get("known_pets", []),
                    "user_timezone": state["chat_settings"].get("user_timezone", "UTC"),
                    "current_pet_species": next(
                        (p.get("species") for p in state.get("known_pets", []) if p.get("name") == state.get("current_pet_name")),
                        ""
                    ),
                }

                # Вызываем агента
                bound_llm = self._bind_llm(state.get("chat_settings"))
                prev_llm = getattr(agent, "llm", None)
                agent.llm = bound_llm

                try:
                    result = await agent.process(
                        user_id=state["user_id"],
                        user_message=agent_message,
                        context=context
                    )
                finally:
                    if prev_llm is not None:
                        agent.llm = prev_llm

                # Сохраняем результат
                if "agent_results" not in state:
                    state["agent_results"] = []

                state["agent_results"].append({
                    "agent": agent_name,
                    "output": result,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                # Извлекаем generated_files если есть
                try:
                    result_data = json.loads(result) if isinstance(result, str) else result
                    if "minio_object_name" in result_data:
                        if "generated_files" not in state:
                            state["generated_files"] = []
                        state["generated_files"].append(result_data)
                except:
                    pass

                logger.info(f"Agent node: {agent_name} completed")

            except Exception as e:
                logger.error(f"Agent node {agent_name} error: {e}")
                if "agent_results" not in state:
                    state["agent_results"] = []
                state["agent_results"].append({
                    "agent": agent_name,
                    "output": f"❌ Ошибка: {str(e)}",
                    "error": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            return state

        return agent_node

    def _build_supervisor_prompt(
        self,
        settings: Dict[str, Any],
        uploaded_files: List[Dict[str, Any]],
        called_agents: List[str],
    ) -> str:
        """Построить system prompt для supervisor"""

        now = datetime.now()

        files_info = ""
        if uploaded_files:
            files_list = [
                f"- {f.get('filename', 'unknown')} ({f.get('file_type', 'unknown')})"
                for f in uploaded_files[:5]
            ]
            files_info = "\n\n**Загруженные файлы:**\n" + "\n".join(files_list)

        called_info = ""
        if called_agents:
            called_info = f"\n\n**Уже вызванные агенты:** {', '.join(called_agents)}"

        settings_info = f"""
**Настройки чата:**
- Веб-поиск: {'✅' if settings.get('web_search_enabled') else '❌'}
- Генерация изображений: {'✅' if settings.get('image_generation_enabled') else '❌'}
- Голосовой ответ: {'✅' if settings.get('voice_response_enabled') else '❌'}
- Модель: {settings.get('gigachat_model', 'GigaChat-Max')}
"""

        prompt = f"""Ты - supervisor мультиагентной системы для владельцев домашних животных.

**Текущие данные:**
- Время: {now.strftime("%Y-%m-%d %H:%M")}
{settings_info}{files_info}{called_info}

**Доступные агенты (8):**

1. **pet_memory** - БД питомцев и медицинские записи
   Когда: упоминание питомца, вопросы о питомцах, медицинские записи

2. **document_rag** - Индексация и поиск в документах
   Когда: загружены документы (PDF, DOCX, TXT, CSV, XLSX), вопросы о документах

3. **multimodal** - Анализ изображений, видео, аудио
   Когда: загружены изображения/видео/аудио, OCR, транскрипция

4. **web_search** - Поиск в интернете (DuckDuckGo)
   Когда: нужна актуальная информация И web_search_enabled=True

5. **health_nutrition** - Анализ здоровья, питания, прививок
   Когда: вопросы о здоровье, питании, расчет норм, анализ корма

6. **calendar** - Google Calendar
   Когда: создание/просмотр событий, запись к ветеринару

7. **content_generation** - Генерация изображений, графиков, аудио (TTS), отчетов
   Когда:
   - Генерация изображений (если image_generation_enabled=True)
   - Генерация голосового ответа / TTS (если voice_response_enabled=True И пользователь просит аудио)
   - Создание графиков, отчетов

8. **email** - Отправка писем по email
   Когда: пользователь просит отправить письмо/уведомление, переслать информацию

**Твоя задача:**
Анализируй запрос и результаты агентов. Решай:
- Нужно ли вызвать ещё агента?
- Или уже можно дать финальный ответ?

**Стратегия:**
- Автоматически сохраняй инфо о питомцах → pet_memory
- Автоматически индексируй загруженные файлы → document_rag / multimodal
- Цепочки агентов:
  * pet_memory → health_nutrition
  * multimodal (OCR) → health_nutrition
  * ЛЮБОЙ агент → content_generation (для TTS) если voice_response_enabled=True И пользователь просит голосовой ответ
- Не вызывай агента повторно если он уже отработал
- Когда есть достаточно данных - формируй финальный ответ

**ВАЖНО про голосовой ответ:**
Если voice_response_enabled=True И пользователь просит "в аудио формате", "голосом", "озвучь", "прочитай вслух":
1. Сначала получи нужную информацию (вызови нужного агента если требуется)
2. Потом вызови content_generation для преобразования ответа в аудио через TTS
3. НЕ давай текстовый ответ, если пользователь просил аудио!"""

        return prompt

    def _build_final_response(self, state: AgentState, fallback_to_llm: bool = False) -> str:
        """Построить финальный ответ на основе результатов агентов. При необходимости — fallback к прямому ответу LLM."""

        agent_results = state.get("agent_results", [])

        response_parts = []

        for result in agent_results:
            if not result.get("error"):
                output = result["output"]
                # Если это JSON, пытаемся извлечь текст
                try:
                    if isinstance(output, str) and output.startswith("{"):
                        data = json.loads(output)
                        if "analysis" in data:
                            response_parts.append(data["analysis"])
                        elif "text" in data:
                            response_parts.append(data["text"])
                        else:
                            response_parts.append(output)
                    else:
                        response_parts.append(output)
                except:
                    response_parts.append(output)

        if response_parts:
            return "\n\n".join(response_parts)

        if fallback_to_llm:
            return self._generate_direct_answer(state)

        return "Ответ не сформирован. Попробуйте ещё раз."

    def _generate_direct_answer(self, state: AgentState) -> str:
        """Финальный ответ напрямую от LLM, если агенты не дали результата."""
        try:
            user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
            last_user_message = user_messages[-1].content if user_messages else ""

            # Добавляем краткий контекст о вызванных агентах/ошибках
            notes = []
            for res in state.get("agent_results", []):
                if res.get("error"):
                    notes.append(f"{res['agent']}: ошибка {res.get('output')}")
            notes_text = "\n".join(notes) if notes else ""

            prompt = f"""Ты ассистент по домашним животным. Дай полезный ответ пользователю.
Запрос: {last_user_message}
{('Предыдущие ошибки агентов:\\n' + notes_text) if notes_text else ''}"""

            llm = self._bind_llm(state.get("chat_settings"))
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Fallback LLM error: {e}")
            return "Не удалось сформировать ответ. Попробуйте позже."

    async def run(
        self,
        messages: List[Message],
        chat_settings: ChatSettingsDTO,
        uploaded_files: List[Dict[str, Any]],
        chat_id: int,
        user_id: int,
    ) -> OrchestratorResult:
        """
        Главный метод оркестратора - запуск LangGraph.

        Args:
            messages: История сообщений чата (объекты Message)
            chat_settings: Настройки чата
            uploaded_files: Загруженные файлы
            chat_id: ID чата
            user_id: ID пользователя

        Returns:
            OrchestratorResult с финальным ответом и метаданными
        """

        logger.info(
            f"Orchestrator (LangGraph) started: user={user_id}, chat={chat_id}, "
            f"messages_count={len(messages)}, files={len(uploaded_files)}"
        )

        try:
            async with self._lock:
                # Конвертируем Message → langchain messages
                lc_messages = self._convert_messages_to_langchain(messages)

                # Инициализируем state
                initial_state: AgentState = {
                    "messages": lc_messages,
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "uploaded_files": uploaded_files,
                    "chat_settings": chat_settings.model_dump(),
                    "current_pet_id": None,
                    "current_pet_name": "",
                    "known_pets": [],
                    "agent_results": [],
                    "generated_files": [],
                    "next_agent": None,
                    "final_response": None,
                }

                # Запускаем граф
                final_state = await self.graph.ainvoke(initial_state)

            # Извлекаем результат
            final_response = final_state.get("final_response", "Обработка завершена")
            agent_results = final_state.get("agent_results", [])
            generated_files = final_state.get("generated_files", [])

            # Метаданные
            metadata = {
                "agents_used": [r["agent"] for r in agent_results if not r.get("error")],
                "total_agents_called": len(agent_results),
                "graph_iterations": len(agent_results) + 1,  # +1 для финального supervisor
            }

            logger.info(
                f"Orchestrator completed: agents={metadata['agents_used']}, "
                f"iterations={metadata['graph_iterations']}, "
                f"generated_files={len(generated_files)}"
            )

            return OrchestratorResult(
                text=final_response,
                metadata=metadata,
                generated_files=generated_files,
            )

        except Exception as e:
            logger.exception(f"Orchestrator error for user {user_id}, chat {chat_id}")
            return OrchestratorResult(
                text=f"Извините, произошла ошибка при обработке запроса: {str(e)}",
                metadata={"error": str(e), "error_type": type(e).__name__},
            )

    def _convert_messages_to_langchain(self, messages: List[Message]) -> List[BaseMessage]:
        """Конвертировать Message в langchain messages"""

        lc_messages = []

        for msg in messages:
            content = msg.content

            # Добавляем информацию о файлах если есть
            if msg.files:
                files_info = "\n\n[Прикрепленные файлы: " + ", ".join(
                    f.get("filename", "unknown") for f in msg.files
                ) + "]"
                content = content + files_info

            if msg.role.value == "user":
                lc_messages.append(HumanMessage(content=content))
            elif msg.role.value == "assistant":
                lc_messages.append(AIMessage(content=content))

        return lc_messages
