# app/agents/orchestrator_agent.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime, timezone
import asyncio
from loguru import logger
import json
import operator

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from app.integrations.gigachat_client import GigaChatClient
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
        llm=None,
    ):
        self.pet_memory_agent = pet_memory_agent
        self.document_rag_agent = document_rag_agent
        self.multimodal_agent = multimodal_agent
        self.web_search_agent = web_search_agent
        self.health_nutrition_agent = health_nutrition_agent
        self.calendar_agent = calendar_agent
        self.content_generation_agent = content_generation_agent

        # Base LLM (env defaults). Per-chat overrides are bound in _bind_llm.
        self._base_llm = llm or GigaChatClient().llm
        self.llm = self._base_llm

        # Prevent concurrent runs from clobbering per-chat LLM binding.
        self._lock = asyncio.Lock()

        # Строим граф
        self.graph = self._build_graph()

        logger.info("OrchestratorAgent (LangGraph) initialized with 7 specialized agents")

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
                END: END,
            }
        )

        return workflow.compile()

    def _bind_llm(self, chat_settings: Dict[str, Any]):
        """
        Возвращает LLM, привязанную к настройкам чата.
        Используем базовую LLM из env и создаем bound-экземпляр без мутации.
        """
        settings = chat_settings or {}
        bind_params: Dict[str, Any] = {}

        if settings.get("gigachat_model"):
            bind_params["model"] = settings["gigachat_model"]

        if settings.get("temperature") is not None:
            bind_params["temperature"] = settings["temperature"]

        if settings.get("max_tokens") is not None:
            bind_params["max_tokens"] = settings["max_tokens"]

        if bind_params:
            return self._base_llm.bind(**bind_params)

        return self._base_llm

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

        # Просим LLM принять решение
        decision_prompt = """Проанализируй ситуацию и реши что делать дальше.

Верни JSON:
{
  "action": "call_agent" | "finish",
  "agent": "pet_memory" | "document_rag" | "multimodal" | "web_search" | "health_nutrition" | "calendar" | "content_generation" | null,
  "reason": "почему это решение",
  "final_response": "итоговый ответ пользователю" (если action=finish)
}

Правила:
- Если нужно сохранить инфо о питомце → call_agent: pet_memory
- Если загружены файлы (документы) → call_agent: document_rag
- Если загружены изображения/видео/аудио → call_agent: multimodal
- Если нужна актуальная инфа И web_search_enabled=True → call_agent: web_search
- Если вопросы о здоровье/питании → call_agent: health_nutrition
- Если создать событие в календаре → call_agent: calendar
- Если сгенерировать контент (изображение/график/аудио) → call_agent: content_generation
- Если уже есть достаточно данных для ответа → finish

Отвечай ТОЛЬКО JSON, без пояснений."""

        context_messages.append(HumanMessage(content=decision_prompt))

        # Вызываем LLM
        llm = self._bind_llm(state.get("chat_settings"))
        response = llm.invoke(context_messages)
        decision_text = response.content if hasattr(response, 'content') else str(response)

        # Парсим JSON
        try:
            # Ищем JSON в ответе
            import re
            json_match = re.search(r'\{.*\}', decision_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(0))
            else:
                decision = {"action": "finish", "reason": "Не удалось распарсить решение"}
        except Exception as e:
            logger.error(f"Failed to parse supervisor decision: {e}")
            decision = {"action": "finish", "reason": f"Ошибка парсинга: {str(e)}"}

        logger.info(f"Supervisor decision: {decision}")

        # Применяем решение
        if decision.get("action") == "call_agent":
            next_agent = decision.get("agent")
            state["next_agent"] = next_agent
            logger.info(f"Supervisor → routing to: {next_agent}")
        else:
            # Формируем финальный ответ
            if decision.get("final_response"):
                state["final_response"] = decision["final_response"]
            else:
                # Если LLM не вернул ответ, формируем сами
                state["final_response"] = self._build_final_response(state)

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

                # Формируем контекст для агента
                context = {
                    "chat_id": state["chat_id"],
                    "uploaded_files": state.get("uploaded_files", []),
                    "chat_settings": state["chat_settings"],
                    "current_pet_id": state.get("current_pet_id"),
                    "current_pet_name": state.get("current_pet_name", ""),
                    "known_pets": state.get("known_pets", []),
                }

                # Вызываем агента
                bound_llm = self._bind_llm(state.get("chat_settings"))
                prev_llm = getattr(agent, "llm", None)
                agent.llm = bound_llm

                try:
                    result = await agent.process(
                        user_id=state["user_id"],
                        user_message=last_user_message,
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
- Модель: {settings.get('gigachat_model', 'GigaChat-Max')}
"""

        prompt = f"""Ты - supervisor мультиагентной системы для владельцев домашних животных.

**Текущие данные:**
- Время: {now.strftime("%Y-%m-%d %H:%M")}
{settings_info}{files_info}{called_info}

**Доступные агенты (7):**

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

7. **content_generation** - Генерация изображений, графиков, аудио, отчетов
   Когда: генерация контента

**Твоя задача:**
Анализируй запрос и результаты агентов. Решай:
- Нужно ли вызвать ещё агента?
- Или уже можно дать финальный ответ?

**Стратегия:**
- Автоматически сохраняй инфо о питомцах → pet_memory
- Автоматически индексируй загруженные файлы → document_rag / multimodal
- Цепочки агентов: pet_memory → health_nutrition, multimodal (OCR) → health_nutrition, и т.д.
- Не вызывай агента повторно если он уже отработал
- Когда есть достаточно данных - формируй финальный ответ"""

        return prompt

    def _build_final_response(self, state: AgentState) -> str:
        """Построить финальный ответ на основе результатов агентов"""

        agent_results = state.get("agent_results", [])

        if not agent_results:
            return "Обработка завершена"

        # Собираем все результаты
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

        return "\n\n".join(response_parts) if response_parts else "Обработка завершена"

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
