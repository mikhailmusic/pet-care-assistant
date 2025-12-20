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
        
        # ВАЖНО: Добавляем исходный запрос пользователя
        # На первой итерации и на последующих - чтобы не забыть составные запросы
        context_messages.append(HumanMessage(content=last_user_message))

        # НОВЫЙ ПРОМПТ ДЛЯ РЕШЕНИЯ
        decision_prompt = self._build_decision_prompt(
            settings=settings_dict,
            called_agents=called_agents,
        )

        # НОВОЕ: Если есть результаты агентов, напоминаем об ИСХОДНОМ запросе
        if called_agents:
            # Проверяем на составные запросы (email + audio, и т.д.)
            compound_indicators = [
                ("аудио", "в виде аудио", "в аудио формате", "озвучь", "голосом"),
                ("email", "на почту", "письмо", "на email")
            ]

            detected_parts = []
            for keywords in compound_indicators:
                if any(kw in last_user_message.lower() for kw in keywords):
                    detected_parts.append(keywords[0])

            reminder = f"\n\n[!] НАПОМИНАНИЕ: Исходный запрос пользователя был:\n\"{last_user_message}\"\n"

            if len(detected_parts) >= 2:
                reminder += f"\n[!] ВНИМАНИЕ: Это СОСТАВНОЙ запрос! Обнаружены части: {', '.join(detected_parts)}\n"
                reminder += f"Уже выполнены агенты: {', '.join(called_agents)}\n"
                reminder += "Проверь, все ли части запроса выполнены! Если нет - вызови нужный агент!\n"

                # Специфичные подсказки
                if "аудио" in detected_parts and "content_generation" not in called_agents:
                    reminder += "[!] Часть про АУДИО ещё НЕ выполнена! Нужно вызвать content_generation для TTS!\n"
                if "email" in detected_parts and "email" not in called_agents:
                    reminder += "[!] Часть про EMAIL ещё НЕ выполнена! Нужно вызвать email агента!\n"
            else:
                reminder += "Проверь, все ли части запроса выполнены!"

            decision_prompt += reminder

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

        # ИСПРАВЛЕНИЕ: Если action = имя агента вместо "call_agent", исправляем
        action = decision.get("action")
        if action and action in self.agents.keys():
            logger.warning(f"Supervisor returned agent name '{action}' as action, fixing to 'call_agent'")
            decision["agent"] = action
            decision["action"] = "call_agent"

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
        if decision.get("action") == "respond":
            # ПРЯМОЙ ОТВЕТ супервизора (для простых запросов)
            message = decision.get("message", "Не удалось обработать запрос.")

            # Добавляем ответ в результаты как будто он от агента
            if "agent_results" not in state:
                state["agent_results"] = []
            state["agent_results"].append({
                "agent": "supervisor",
                "output": message,
                "error": False,
            })

            state["next_agent"] = "finalize"
            logger.info(f"Supervisor → direct response: {message[:50]}...")

        elif decision.get("action") == "call_agent":
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
                    # Для веб-поиска - формируем ответ на основе загруженных страниц
                    elif "search_results" in data and "loaded_pages" in data:
                        # Получаем данные
                        search_results = data.get("search_results", [])
                        loaded_pages = data.get("loaded_pages", [])
                        summary = data.get("summary", "")

                        # Формируем ответ на основе loaded_pages
                        if loaded_pages:
                            # Используем LLM для создания качественного ответа
                            user_query = state.get("messages", [])[-1].content if state.get("messages") else ""

                            # Собираем контент со всех загруженных страниц
                            sources_content = []
                            for idx, page in enumerate(loaded_pages[:3], 1):
                                sources_content.append(
                                    f"### Источник {idx}: {page.get('title', 'Без названия')}\n"
                                    f"URL: {page.get('url', '')}\n\n"
                                    f"{page.get('content', '')[:5000]}\n"
                                )

                            combined_content = "\n\n---\n\n".join(sources_content)

                            # Формируем промпт для LLM
                            synthesis_prompt = f"""Пользователь задал вопрос: "{user_query}"

Ты получил следующую информацию из интернета:

{combined_content}

Твоя задача:
1. Проанализируй информацию из всех источников
2. Создай структурированный, понятный ответ на вопрос пользователя
3. Выдели ключевые моменты и практические рекомендации
4. НЕ копируй текст напрямую - переформулируй и структурируй
5. В конце добавь раздел "Источники:" со ссылками

Формат ответа:
- Используй маркированные списки для структурирования
- Выделяй важную информацию
- Пиши понятным языком
- Обязательно добавь ссылки на источники в конце"""

                            try:
                                # Используем LLM для синтеза ответа
                                from langchain_core.messages import HumanMessage
                                synthesis_result = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
                                synthesized_answer = synthesis_result.content

                                # Добавляем ссылки на источники, если их нет
                                if "Источники:" not in synthesized_answer and search_results:
                                    sources_text = "\n\nИсточники:"
                                    for sr in search_results[:5]:
                                        sources_text += f"\n- [{sr.get('title', 'Без названия')}]({sr.get('url', '')})"
                                    synthesized_answer += sources_text

                                response_parts.append(synthesized_answer)

                            except Exception as e:
                                logger.error(f"Failed to synthesize answer from loaded pages: {e}")
                                # Fallback: показываем краткую информацию
                                fallback_parts = []
                                if summary:
                                    fallback_parts.append(summary)

                                fallback_parts.append(f"\nНайдено {len(search_results)} результатов. Вот краткая информация:")

                                for page in loaded_pages[:2]:
                                    fallback_parts.append(f"\n**{page.get('title', 'Без названия')}**")
                                    fallback_parts.append(page.get('content', '')[:500] + "...")

                                if search_results:
                                    sources_text = "\n\nИсточники:"
                                    for sr in search_results[:5]:
                                        sources_text += f"\n- [{sr.get('title', 'Без названия')}]({sr.get('url', '')})"
                                    fallback_parts.append(sources_text)

                                response_parts.append("\n".join(fallback_parts))
                        else:
                            # Если нет загруженных страниц, показываем результаты поиска
                            if search_results:
                                search_text = f"Найдено {len(search_results)} результатов:\n\n"
                                for sr in search_results[:5]:
                                    search_text += f"- [{sr.get('title', 'Без названия')}]({sr.get('url', '')})\n  {sr.get('snippet', '')}\n\n"
                                response_parts.append(search_text)
                    # Для старого формата с analysis (обратная совместимость)
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

            # НОВОЕ: Если LLM вернул несколько JSON объектов, берём только ПЕРВЫЙ
            text = decision_text.strip()

            # Находим первый { и соответствующую ему }
            if "{" in text:
                start_idx = text.find("{")
                # Используем счётчик скобок для поиска конца ПЕРВОГО JSON объекта
                brace_count = 0
                end_idx = -1

                for i in range(start_idx, len(text)):
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break

                if end_idx > start_idx:
                    first_json = text[start_idx:end_idx]
                    decision = json.loads(first_json)

                    # Проверяем, был ли это случай с несколькими JSON
                    remaining_text = text[end_idx:].strip()
                    if remaining_text and remaining_text.startswith("{"):
                        logger.warning(
                            f"LLM returned multiple JSON objects! Using only the FIRST one. "
                            f"First: {first_json[:100]}, Remaining: {remaining_text[:100]}"
                        )

                    return decision

            # Fallback: пробуем парсить весь текст как есть
            decision = json.loads(text)
            return decision

        except json.JSONDecodeError as e:
            # Показываем фрагмент текста вокруг ошибки для диагностики
            start = max(0, e.pos - 50)
            end = min(len(decision_text), e.pos + 50)
            context = decision_text[start:end]
            logger.error(
                f"Failed to parse decision as JSON: {e}\n"
                f"Position: {e.pos}, Line: {e.lineno}, Column: {e.colno}\n"
                f"Context around error: ...{context}...\n"
                f"Full text preview: {decision_text[:200]}"
            )
            return {"action": "finish", "reason": "parse error"}
        except Exception as e:
            logger.error(f"Failed to parse decision: {e}, text: {decision_text[:200]}")
            return {"action": "finish", "reason": "parse error"}
    
    def _create_agent_node(self, agent_name: str, agent):
        """Создать узел для агента"""
        
        async def agent_node(state: AgentState) -> AgentState:
            logger.info(f"Agent node: {agent_name} started")
            
            try:
                user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
                last_user_message = user_messages[-1].content if user_messages else ""
                
                # Обогащаем сообщение для content_generation если нужен TTS
                # или для email agent если нужно отправить предыдущий ответ
                agent_message = last_user_message
                agent_results = state.get("agent_results", [])

                # Проверяем, просит ли пользователь явно создать аудио
                tts_keywords = ["аудио", "озвучь", "голосом", "в виде аудио", "audio", "tts", "прочитай вслух", "аудиоверс"]
                user_wants_audio = any(keyword in last_user_message.lower() for keyword in tts_keywords)

                # Проверяем, просит ли пользователь отправить последний ответ на email
                email_last_response_keywords = ["последний ответ", "твой ответ", "этот ответ", "твой последний", "предыдущий ответ"]
                user_wants_last_response = any(keyword in last_user_message.lower() for keyword in email_last_response_keywords)

                if agent_name == "content_generation" and user_wants_audio:
                    # Собираем текст для озвучивания
                    text_to_synthesize = None

                    # Случай 1: Есть результаты от других агентов - озвучиваем их
                    if agent_results:
                        previous_texts = []

                        for res in agent_results:
                            if not res.get("error"):
                                agent_who_ran = res.get("agent", "")
                                output = res["output"]

                                # Специальная обработка для email агента
                                if agent_who_ran == "email":
                                    try:
                                        # Обрабатываем токены GigaChat перед парсингом
                                        if isinstance(output, str):
                                            cleaned = output.replace("<|superquote|>", '"')

                                            # Экранируем переносы строк в строковых значениях
                                            import re
                                            def escape_newlines_in_strings(match):
                                                value = match.group(1)
                                                value = value.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                                                return f'"{value}"'

                                            cleaned = re.sub(r'"([^"]*)"', escape_newlines_in_strings, cleaned, flags=re.DOTALL)
                                            data = json.loads(cleaned)
                                        else:
                                            data = output

                                        if data.get("email_sent"):
                                            # Формируем подтверждение об отправке письма
                                            recipient = data.get("recipient_email", "")
                                            subject = data.get("subject", "")
                                            confirmation = f"Письмо успешно отправлено на {recipient} с темой \"{subject}\""
                                            previous_texts.append(confirmation)
                                            logger.info(f"TTS: prepared email confirmation: {confirmation}")
                                            continue
                                    except Exception as e:
                                        logger.warning(f"Failed to parse email result for TTS: {e}")

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
                        logger.info(f"TTS: preparing text (length={len(text_to_synthesize)}): {text_to_synthesize[:200]}...")
                        agent_message = f"""Вызови инструмент text_to_speech со следующим текстом.

ТЕКСТ ДЛЯ ОЗВУЧИВАНИЯ (передай его полностью в параметр text):
{text_to_synthesize}

Параметры для text_to_speech:
- text: (весь текст выше)
- voice: May_24000
- audio_format: wav16

КРИТИЧЕСКИ ВАЖНО: Используй ВЕСЬ текст выше (от первого до последнего символа) в параметре 'text' при вызове text_to_speech. Верни ТОЛЬКО JSON результат от инструмента."""

                # Обогащаем для email agent если нужно отправить последний ответ
                elif agent_name == "email" and user_wants_last_response:
                    # Находим последнее сообщение ассистента
                    text_to_send = None

                    # Случай 1: Есть результаты от других агентов
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
                                            previous_texts.append(json.dumps(data, ensure_ascii=False, indent=2))
                                    else:
                                        previous_texts.append(output)
                                except:
                                    previous_texts.append(output)

                        if previous_texts:
                            text_to_send = "\n\n".join(previous_texts)

                    # Случай 2: Нет результатов агентов - ищем последнее сообщение ассистента
                    if not text_to_send:
                        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
                        if ai_messages:
                            text_to_send = ai_messages[-1].content

                    # Если нашли текст - добавляем в сообщение
                    if text_to_send:
                        agent_message = f"""{last_user_message}

КОНТЕКСТ: Пользователь просит отправить последний ответ на email.
Последний ответ ассистента:
---
{text_to_send}
---

Используй этот текст как body письма. Сформулируй подходящую тему (subject) на основе содержания."""

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
                    # Для email агента добавляем историю разговора
                    if agent_name == "email":
                        # Конвертируем langchain messages в простой формат для email агента
                        conversation_history = []
                        for msg in state.get("messages", []):
                            if hasattr(msg, "type"):
                                role = "user" if msg.type == "human" else "assistant"
                                conversation_history.append({
                                    "role": role,
                                    "content": msg.content
                                })

                        result = await agent.process(
                            user_id=state["user_id"],
                            user_message=agent_message,
                            context=context,
                            conversation_history=conversation_history
                        )
                    else:
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
                    # Обрабатываем токены GigaChat перед парсингом
                    if isinstance(result, str):
                        cleaned_result = result.replace("<|superquote|>", '"')
                        import re
                        cleaned_result = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned_result)
                        result_data = json.loads(cleaned_result)
                    else:
                        result_data = result

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
