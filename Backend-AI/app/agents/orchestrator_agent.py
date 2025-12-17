from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timezone
from loguru import logger
import json
import asyncio

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from app.dto import ChatSettingsDTO
from app.models.message import Message
from app.config import settings
from app.agents.orchestrator_prompts import (
    build_supervisor_system_prompt,
    build_decision_prompt,
)


@dataclass
class OrchestratorResult:
    text: str
    metadata: Dict[str, Any]
    generated_files: List[Dict[str, Any]] = None


class AgentState(Dict):
    messages: List[BaseMessage]
    user_id: int
    chat_id: int
    uploaded_files: List[Dict[str, Any]]
    chat_settings: Dict[str, Any]
    current_pet_id: Optional[int]
    current_pet_name: str
    known_pets: List[Dict[str, Any]]
    agent_results: List[Dict[str, Any]]
    generated_files: List[Dict[str, Any]]
    next_agent: Optional[str]
    final_response: Optional[str]
    shared_context: Dict[str, Any]


class OrchestratorAgent:
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
        llm,
        llm_factory,
        max_iterations: int = 10,
    ):
        self.agents = {
            "pet_memory": pet_memory_agent,
            "document_rag": document_rag_agent,
            "multimodal": multimodal_agent,
            "web_search": web_search_agent,
            "health_nutrition": health_nutrition_agent,
            "calendar": calendar_agent,
            "content_generation": content_generation_agent,
            "email": email_agent,
        }
        
        self.llm = llm
        self._llm_factory = llm_factory
        self.max_iterations = max_iterations
        self._lock = asyncio.Lock()
        
        self.graph = self._create_graph()
        
        logger.info(f"OrchestratorAgent initialized with {len(self.agents)} agents")
    
    def _create_graph(self) -> StateGraph:
        
        workflow = StateGraph(AgentState)
        
        # Добавляем узлы
        workflow.add_node("supervisor", self._supervisor_node)
        
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, self._create_agent_node(agent_name, agent))
        
        workflow.add_node("finalize", self._finalize_response_node)
        
        # Точка входа
        workflow.set_entry_point("supervisor")
        
        # Conditional edges от supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_next,
            {
                "pet_memory": "pet_memory",
                "document_rag": "document_rag",
                "multimodal": "multimodal",
                "web_search": "web_search",
                "health_nutrition": "health_nutrition",
                "calendar": "calendar",
                "content_generation": "content_generation",
                "email": "email",
                "finalize": "finalize", 
            }
        )
        
        # От каждого агента обратно в supervisor
        for agent_name in self.agents.keys():
            workflow.add_edge(agent_name, "supervisor")
        
        # От finalize → END
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _route_next(self, state: AgentState) -> str:
        next_agent = state.get("next_agent")
        
        if next_agent == END or next_agent == "finalize":
            return "finalize"
        
        return next_agent or "finalize"
    
    def _bind_llm(self, chat_settings: Optional[Dict[str, Any]]):
        """Создать LLM с учётом настроек"""
        if not chat_settings:
            return self.llm
        
        model_name = chat_settings.get("gigachat_model")
        if not model_name or model_name == settings.GIGACHAT_MODEL:
            return self.llm
        
        return self._llm_factory(chat_settings=chat_settings)
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        logger.info(f"Supervisor: analyzing state (user={state['user_id']}, chat={state['chat_id']})")
        
        user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        last_user_message = user_messages[-1].content if user_messages else ""
        
        called_agents = [r["agent"] for r in state.get("agent_results", [])]
        
        # Предохранитель
        if len(called_agents) >= self.max_iterations:
            logger.warning(f"Max iterations reached ({self.max_iterations}), finishing")
            state["next_agent"] = "finalize"
            return state
        
        settings_dict = state["chat_settings"]
        uploaded_files = state.get("uploaded_files", [])
        
        system_prompt = self._build_supervisor_prompt(
            settings=settings_dict,
            uploaded_files=uploaded_files,
            called_agents=called_agents,
            shared_context=state.get("shared_context", {}),
        )
        
        context_messages = [SystemMessage(content=system_prompt)]
        
        # Добавляем результаты агентов
        for result in state.get("agent_results", []):
            agent_name = result["agent"]
            agent_output = result["output"]
            
            # Извлекаем ключевую информацию из JSON если есть
            summary = self._extract_result_summary(agent_output)
            
            context_messages.append(
                AIMessage(content=f"[{agent_name}] {summary}")
            )
        
        context_messages.append(HumanMessage(content=last_user_message))
        
        # НОВЫЙ ПРОМПТ ДЛЯ РЕШЕНИЯ
        decision_prompt = self._build_decision_prompt(
            settings=settings_dict,
            called_agents=called_agents,
        )
        
        context_messages.append(HumanMessage(content=decision_prompt))
        
        llm = self._bind_llm(state.get("chat_settings"))
        try:
            response = llm.invoke(context_messages)
            decision_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Supervisor LLM error: {e}")
            state["next_agent"] = "finalize"
            # Добавляем ошибку в результаты для финального узла
            if "agent_results" not in state:
                state["agent_results"] = []
            state["agent_results"].append({
                "agent": "supervisor",
                "output": "Модель недоступна",
                "error": True,
            })
            return state
        
        decision = self._parse_decision(decision_text)
        logger.info(f"Supervisor decision: {decision}")
        
        # Проверка разрешённых функций
        agent = decision.get("agent") if decision.get("action") == "call_agent" else None
        
        if agent == "web_search" and not settings_dict.get("web_search_enabled", False):
            state["next_agent"] = "finalize"
            if "agent_results" not in state:
                state["agent_results"] = []
            state["agent_results"].append({
                "agent": "supervisor",
                "output": "В этом чате отключён веб-поиск. Могу ответить без интернета, либо включи веб-поиск в настройках.",
                "error": False,
            })
            return state
        
        if agent == "content_generation":
            if not settings_dict.get("image_generation_enabled", False) and not settings_dict.get("voice_response_enabled", False):
                state["next_agent"] = "finalize"
                if "agent_results" not in state:
                    state["agent_results"] = []
                state["agent_results"].append({
                    "agent": "supervisor",
                    "output": "В этом чате отключена генерация контента и голосовой ответ. Включи нужные функции в настройках.",
                    "error": False,
                })
                return state
        
        # Применяем решение
        if decision.get("action") == "call_agent":
            next_agent = decision.get("agent")
            
            # Предотвращаем повторные вызовы
            if next_agent in called_agents:
                logger.warning(f"Agent {next_agent} already called, finishing")
                state["next_agent"] = "finalize"
                return state
            
            state["next_agent"] = next_agent
            
            # НОВОЕ: сохраняем контекст для следующего агента
            if decision.get("context_note"):
                if "shared_context" not in state:
                    state["shared_context"] = {}
                state["shared_context"]["last_note"] = decision.get("context_note")
            
            logger.info(f"Supervisor → routing to: {next_agent}")
        else:
            # Завершаем работу
            state["next_agent"] = "finalize"
            logger.info(f"Supervisor → finalize")
        
        return state
    
    def _extract_result_summary(self, output: str) -> str:
        try:
            if isinstance(output, str) and output.startswith("{"):
                data = json.loads(output)
                
                if "email_sent" in data:
                    email = data.get("recipient_email", "unknown")
                    return f"Письмо отправлено на {email}"
                
                # Для TTS
                if "minio_url" in data and "text_preview" in data:
                    return f"Создан аудиофайл: {data.get('minio_url')}"
                
                # Для других агентов
                if "analysis" in data:
                    preview = data["analysis"][:200]
                    return f"{preview}..."
                
                if "text" in data:
                    preview = data["text"][:200]
                    return f"{preview}..."
                
                return str(data)[:200]
            else:
                return output[:200]
        except:
            return output[:200]
    
    def _finalize_response_node(self, state: AgentState) -> AgentState:
        """
        НОВЫЙ УЗЕЛ: умная финализация ответа
        
        Агрегирует результаты всех агентов без дублирования
        и формирует единый связный ответ для пользователя
        """
        
        logger.info("Finalize: building final response")
        
        agent_results = state.get("agent_results", [])
        shared_context = state.get("shared_context", {})
        
        # Группируем результаты по типам
        info_results = []
        action_results = []
        error_results = []
        
        for result in agent_results:
            if result.get("error"):
                error_results.append(result)
            elif result["agent"] in ["email", "calendar"]:
                # Действия (отправка email, создание события)
                action_results.append(result)
            else:
                # Информационные результаты
                info_results.append(result)
        
        # Формируем ответ
        response_parts = []
        
        # 1. Действия (если были)
        if action_results:
            for result in action_results:
                output = result["output"]
                
                # Извлекаем читаемый текст
                try:
                    if isinstance(output, str) and output.startswith("{"):
                        data = json.loads(output)
                        
                        if "email_sent" in data:
                            email = data.get("recipient_email")
                            response_parts.append(f"✅ Письмо успешно отправлено на {email}")
                        elif "event_created" in data:
                            response_parts.append(f"✅ Событие создано в календаре")
                        else:
                            response_parts.append(output)
                    else:
                        response_parts.append(output)
                except:
                    response_parts.append(output)
        
        # 2. Информация (без дублирования)
        seen_content = set()
        
        for result in info_results:
            output = result["output"]
            
            # Проверяем на дубликаты по первым 100 символам
            content_hash = output[:100] if len(output) > 100 else output
            
            if content_hash in seen_content:
                continue
            
            seen_content.add(content_hash)
            
            # Извлекаем текст
            try:
                if isinstance(output, str) and output.startswith("{"):
                    data = json.loads(output)

                    # Для TTS/изображений/отчетов - формируем информативное сообщение
                    # Файлы уже в generated_files и будут переданы во фронтенд через metadata
                    if "minio_url" in data:
                        if "synthesized_at" in data:
                            # Для аудио - пропускаем, т.к. фронтенд показывает аудиоплеер
                            continue
                        elif "generated_at" in data and "prompt" in data:
                            # Для изображений - пропускаем, т.к. фронтенд показывает изображение
                            continue
                        elif "created_at" in data and "title" in data:
                            # Для отчётов - пропускаем, т.к. фронтенд показывает ссылку на скачивание
                            continue
                    # Для других JSON результатов
                    elif "analysis" in data:
                        response_parts.append(data["analysis"])
                    elif "text" in data:
                        response_parts.append(data["text"])
                    else:
                        # Оставляем как есть, если не можем извлечь
                        response_parts.append(output)
                else:
                    response_parts.append(output)
            except:
                response_parts.append(output)
        
        # 3. Ошибки (если были)
        if error_results:
            errors = [f"⚠️ {r['output']}" for r in error_results]
            response_parts.extend(errors)
        
        # 4. Если нет результатов вообще
        if not response_parts:
            # Проверяем, есть ли generated_files - если есть, добавляем информативное сообщение
            generated_files = state.get("generated_files", [])
            if generated_files:
                # Формируем сообщение в зависимости от типа файлов
                file_types = []
                for gf in generated_files:
                    if "synthesized_at" in gf:
                        file_types.append("аудиофайл")
                    elif "generated_at" in gf and "prompt" in gf:
                        file_types.append("изображение")
                    elif "created_at" in gf and "title" in gf:
                        file_types.append("отчёт")
                    elif "chart_type" in gf:
                        file_types.append("график")

                if file_types:
                    files_text = ", ".join(set(file_types))
                    response_parts.append(f"Готово! Создан {files_text}.")
                else:
                    response_parts.append("Файл успешно создан.")
            else:
                response_parts.append("Не удалось обработать запрос. Попробуйте переформулировать.")

        # Объединяем без дублирования
        final_response = "\n\n".join(response_parts)
        
        # Дедупликация (удаляем полностью одинаковые абзацы)
        paragraphs = final_response.split("\n\n")
        unique_paragraphs = []
        seen_paragraphs = set()
        
        for p in paragraphs:
            if p not in seen_paragraphs:
                unique_paragraphs.append(p)
                seen_paragraphs.add(p)
        
        state["final_response"] = "\n\n".join(unique_paragraphs)
        state["next_agent"] = END
        
        logger.info(f"Finalize: response built, length={len(state['final_response'])}")
        
        return state
    
    def _build_supervisor_prompt(
        self,
        settings: Dict[str, Any],
        uploaded_files: List[Dict[str, Any]],
        called_agents: List[str],
        shared_context: Dict[str, Any],
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

        context_info = ""
        if shared_context:
            context_notes = []
            if "last_note" in shared_context:
                context_notes.append(f"- {shared_context['last_note']}")


            if context_notes:
                context_info = "\n\n**Контекст из предыдущих действий:**\n" + "\n".join(context_notes)

        settings_info = f"""
**Настройки чата:**
- Веб-поиск: {'✅' if settings.get('web_search_enabled') else '❌'}
- Генерация изображений: {'✅' if settings.get('image_generation_enabled') else '❌'}
- Голосовой ответ (авто): {'✅' if settings.get('voice_response_enabled') else '❌'}
- Модель: {settings.get('gigachat_model', 'GigaChat-Max')}
"""

        # Используем промпт из отдельного модуля
        return build_supervisor_system_prompt(
            now=now,
            settings_info=settings_info,
            files_info=files_info,
            called_info=called_info,
            context_info=context_info,
        )

    def _build_decision_prompt(
        self,
        settings: Dict[str, Any],
        called_agents: List[str],
    ) -> str:
        """Построить промпт для принятия решения"""

        enabled_features = []
        disabled_features = []

        if settings.get("web_search_enabled"):
            enabled_features.append("веб-поиск")
        else:
            disabled_features.append("web_search")

        if settings.get("image_generation_enabled"):
            enabled_features.append("генерация изображений")
        else:
            disabled_features.append("content_generation (изображения)")

        if settings.get("voice_response_enabled"):
            enabled_features.append("голосовой ответ (TTS)")

        # Используем промпт из отдельного модуля
        return build_decision_prompt(
            enabled_features=enabled_features,
            disabled_features=disabled_features,
        )
    
    def _parse_decision(self, decision_text: str) -> Dict[str, Any]:
        """Парсинг JSON решения от LLM"""
        try:
            # Убираем markdown если есть
            if "```json" in decision_text:
                decision_text = decision_text.split("```json")[1].split("```")[0]
            elif "```" in decision_text:
                decision_text = decision_text.split("```")[1].split("```")[0]
            
            decision = json.loads(decision_text.strip())
            return decision
        except Exception as e:
            logger.error(f"Failed to parse decision: {e}, text: {decision_text}")
            return {"action": "finish", "reason": "parse error"}
    
    def _create_agent_node(self, agent_name: str, agent):
        """Создать узел для агента"""
        
        async def agent_node(state: AgentState) -> AgentState:
            logger.info(f"Agent node: {agent_name} started")
            
            try:
                user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
                last_user_message = user_messages[-1].content if user_messages else ""
                
                # Обогащаем сообщение для content_generation если нужен TTS
                agent_message = last_user_message
                agent_results = state.get("agent_results", [])

                # Проверяем, просит ли пользователь явно создать аудио
                tts_keywords = ["аудио", "озвучь", "голосом", "в виде аудио", "audio", "tts", "прочитай вслух", "аудиоверс"]
                user_wants_audio = any(keyword in last_user_message.lower() for keyword in tts_keywords)

                if agent_name == "content_generation" and user_wants_audio:
                    # Собираем текст для озвучивания
                    text_to_synthesize = None

                    # Случай 1: Есть результаты от других агентов - озвучиваем их
                    if agent_results:
                        previous_texts = []

                        for res in agent_results:
                            if not res.get("error"):
                                output = res["output"]

                                try:
                                    if isinstance(output, str) and output.startswith("{"):
                                        data = json.loads(output)

                                        if "analysis" in data:
                                            previous_texts.append(data["analysis"])
                                        elif "text" in data:
                                            previous_texts.append(data["text"])
                                        else:
                                            # Для других JSON результатов берём весь JSON как строку
                                            previous_texts.append(json.dumps(data, ensure_ascii=False, indent=2))
                                    else:
                                        previous_texts.append(output)
                                except:
                                    previous_texts.append(output)

                        if previous_texts:
                            text_to_synthesize = "\n\n".join(previous_texts)

                    # Случай 2: Нет результатов агентов - ищем предыдущее сообщение ассистента
                    if not text_to_synthesize:
                        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
                        if ai_messages:
                            # Берём последнее сообщение ассистента
                            text_to_synthesize = ai_messages[-1].content

                    # Если нашли текст для озвучивания - формируем промпт
                    if text_to_synthesize:
                        agent_message = f"""ВАЖНО: Используй инструмент text_to_speech для создания аудиофайла!

Текст для озвучивания:
{text_to_synthesize}

Параметры:
- voice: May_24000
- audio_format: wav16

Вызови text_to_speech с этим текстом и верни ТОЛЬКО JSON результат от инструмента."""
                
                # Формируем контекст
                context = {
                    "chat_id": state["chat_id"],
                    "uploaded_files": state.get("uploaded_files", []),
                    "chat_settings": state["chat_settings"],
                    "current_pet_id": state.get("current_pet_id"),
                    "current_pet_name": state.get("current_pet_name", ""),
                    "known_pets": state.get("known_pets", []),
                    "user_timezone": state["chat_settings"].get("user_timezone") or settings.DEFAULT_TIMEZONE,
                    "current_pet_species": next(
                        (p.get("species") for p in state.get("known_pets", []) 
                         if p.get("name") == state.get("current_pet_name")),
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

                # НОВОЕ: сохраняем важную информацию в shared_context
                try:
                    result_data = json.loads(result) if isinstance(result, str) else result

                    # Для email агента сохраняем email
                    if agent_name == "email" and "recipient_email" in result_data:
                        if "shared_context" not in state:
                            state["shared_context"] = {}
                        state["shared_context"]["last_email"] = result_data["recipient_email"]

                    # Извлекаем generated_files
                    if "minio_object_name" in result_data:
                        if "generated_files" not in state:
                            state["generated_files"] = []
                        state["generated_files"].append(result_data)
                        logger.info(f"Added file to generated_files: {result_data.get('minio_object_name')}")
                except Exception as e:
                    logger.warning(f"Could not parse result as JSON for agent {agent_name}: {e}, result preview: {str(result)[:200]}")
                
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
    
    async def run(
        self,
        messages: List[Message],
        chat_settings: ChatSettingsDTO,
        uploaded_files: List[Dict[str, Any]],
        chat_id: int,
        user_id: int,
    ) -> OrchestratorResult:
        """Главный метод оркестратора - запуск LangGraph"""
        
        logger.info(
            f"Orchestrator (FIXED) started: user={user_id}, chat={chat_id}, "
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
                    "shared_context": {},  # НОВОЕ
                }
                
                # Запускаем граф
                final_state = await self.graph.ainvoke(initial_state)
            
            final_response = final_state.get("final_response", "Обработка завершена")
            agent_results = final_state.get("agent_results", [])
            generated_files = final_state.get("generated_files", [])
            
            metadata = {
                "agents_used": [r["agent"] for r in agent_results if not r.get("error")],
                "total_agents_called": len(agent_results),
                "graph_iterations": len(agent_results) + 1,
            }
            
            logger.info(
                f"Orchestrator (FIXED) completed: agents={metadata['agents_used']}, "
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
