
from __future__ import annotations

from typing import Literal, Optional, Dict, Any, Sequence
from loguru import logger

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState
from app.agents.agent_factory import AgentFactory


# ============================================================================
# SUPERVISOR NODE
# ============================================================================

def create_supervisor_node(llm, handoff_tools: list, chat_settings: Optional[Dict] = None):
    """Создать supervisor node с учётом настроек"""
    
    settings = chat_settings or {}
    web_search_enabled = settings.get("web_search_enabled", True)
    image_generation_enabled = settings.get("image_generation_enabled", True)
    
    # Формируем список доступных агентов
    available_agents = [tool.name.replace("transfer_to_", "") for tool in handoff_tools]
    
    # Добавляем предупреждения о недоступных функциях
    restrictions = []
    
    if not web_search_enabled:
        restrictions.append(
            "⚠️ **Поиск в интернете ОТКЛЮЧЁН пользователем**. "
            "Если пользователь явно просит найти информацию в интернете - "
            "вежливо откажи и объясни что функция отключена в настройках чата."
        )
    
    if not image_generation_enabled:
        restrictions.append(
            "⚠️ **Генерация контента ОТКЛЮЧЕНА пользователем**. "
            "Если пользователь просит создать изображение, график, отчёт или аудио - "
            "вежливо откажи и объясни что функция отключена в настройках чата."
        )
    
    restrictions_text = "\n".join(restrictions) if restrictions else ""
    
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    supervisor_prompt = f"""Ты - координатор мульти-агентной системы для помощи владельцам домашних животных.

Твоя задача - анализировать запрос пользователя и передавать управление специализированным агентам.

**Доступные агенты в этом чате:**
{", ".join(available_agents)}

{restrictions_text}

**Описание агентов:**

1. **pet_memory** - Управление данными о питомцах (ВСЕГДА ДОСТУПЕН)
   - Создание/обновление профилей питомцев
   - Ведение медицинских записей (прививки, анализы, посещения врача)
   - Получение информации о питомцах

2. **web_search** - Поиск информации в интернете {'(ДОСТУПЕН)' if web_search_enabled else '(ОТКЛЮЧЁН)'}
   - Актуальная информация, новости, факты
   - Используй ТОЛЬКО если:
     * Пользователь ЯВНО просит найти в интернете
     * Или информация точно требует актуальных данных (курсы валют, погода, новости)
   - АВТОМАТИЧЕСКИ НЕ ИСПОЛЬЗУЙ без явного запроса

3. **email** - Отправка email (ВСЕГДА ДОСТУПЕН)
   - Отправка писем на указанный email

4. **document_rag** - Работа с документами (ВСЕГДА ДОСТУПЕН)
   - Индексация загруженных документов (PDF, DOCX, TXT, etc)
   - Поиск информации в документах пользователя
   - Индексация длинного текста из сообщений

5. **content_generation** - Генерация контента {'(ДОСТУПЕН)' if image_generation_enabled else '(ОТКЛЮЧЁН)'}
   - Создание изображений (GigaChat)
   - Построение графиков и диаграмм (matplotlib)
   - Синтез речи (TTS)
   - Генерация отчётов (PDF, DOCX)

6. **multimodal** - Анализ мультимедиа (ВСЕГДА ДОСТУПЕН)
   - Анализ изображений (описание, идентификация породы)
   - OCR - извлечение текста из изображений
   - Транскрибация аудио в текст
   - Анализ видео

7. **health_nutrition** - Здоровье и питание (ВСЕГДА ДОСТУПЕН)
   - Анализ медицинских записей за период
   - Расчёт суточной нормы питания
   - Анализ состава корма
   - Проверка графика прививок

8. **calendar** - Google Calendar (если подключён)
   - Создание/изменение/удаление событий
   - Проверка занятости
   - Напоминания

**Правила работы:**

1. **Уважай настройки пользователя:**
   - Если функция отключена - НЕ используй агента, ОТКАЖИ вежливо
   - Объясни что функцию можно включить в настройках чата

2. **Web Search - только по явному запросу или для актуальной информации. Состояние на данный момент: {'ДОСТУПЕН' if web_search_enabled else 'ОТКЛЮЧЁН'}:**
   - "Найди в интернете..." → OK, используй web_search
   - "Какая погода?" → OK, используй web_search (актуальные данные)
   - "Расскажи о породе кошек" → НЕТ, отвечай сам из знаний
   - "Какой корм лучше?" → НЕТ, отвечай сам, не ищи автоматически

3. **Content Generation - только по явному запросу. Состояние на данный момент: {'ДОСТУПЕН' if image_generation_enabled else 'ОТКЛЮЧЁН'}:**
   - "Создай картинку..." → OK (если включено)
   - "Построй график..." → OK (если включено)
   - Обычный вопрос → НЕТ, не генерируй контент без запроса

4. **Многошаговые задачи:**
   - Можешь вызывать несколько агентов последовательно
   - Например: web_search → content_generation (найти информацию, потом создать отчёт)
   - Или: pet_memory → health_nutrition (получить данные питомца, потом рассчитать питание)

5. **Когда закончить:**
   - Если ты можешь ответить сам (простой вопрос) - ответь и не вызывай агентов
   - Если все агенты выполнили работу и есть финальный ответ - закончи
   - НЕ вызывай агентов без необходимости

6. **Формирование ответа:**
   - Используй результаты работы агентов для формирования ответа
   - Если агент вернул JSON - извлеки из него нужную информацию
   - Отвечай понятным языком, структурируй информацию

**Важно:**
- Всегда используй transfer_to_X для передачи управления агенту
- Не пытайся сам выполнить работу агента
- Если агент вернул ошибку - сообщи пользователю

Сейчас: {current_time}
"""
    
    supervisor_agent = create_react_agent(
        model=llm,
        tools=handoff_tools,
        prompt=supervisor_prompt,
    )
    
    async def supervisor_node(state: AgentState) -> AgentState:
        """Supervisor node - принимает решение куда передать управление"""
        logger.info("=== SUPERVISOR NODE ===")
        
        # Вызываем supervisor agent
        result = await supervisor_agent.ainvoke(state)
        
        # Извлекаем последнее сообщение
        last_message = result["messages"][-1]
        
        # Проверяем есть ли tool calls
        next_agent = None
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Берём первый tool call
            tool_call = last_message.tool_calls[0]
            tool_name = tool_call["name"]
            
            # Извлекаем имя агента из tool name
            if tool_name.startswith("transfer_to_"):
                next_agent = tool_name.replace("transfer_to_", "")
                logger.info(f"Supervisor decided to transfer to: {next_agent}")
            else:
                logger.warning(f"Unknown tool call: {tool_name}")
        else:
            # Нет tool calls - значит supervisor хочет закончить
            logger.info("Supervisor decided to finish (no tool calls)")
        
        # Обновляем state
        return {
            **state,
            "messages": result["messages"],
            "next_agent": next_agent,
        }
    
    return supervisor_node


# ============================================================================
# AGENT NODE WRAPPER
# ============================================================================

def create_agent_node(agent_name: str):
    """Создать node для конкретного агента
    
    Args:
        agent_name: Имя агента (например, "pet_memory")
    
    Returns:
        Функция agent node
    """
    
    async def agent_node(state: AgentState) -> AgentState:
        """Agent node - вызывает конкретного агента
        
        Args:
            state: Текущее состояние
        
        Returns:
            Обновлённое состояние с результатом работы агента
        """
        logger.info(f"=== AGENT NODE: {agent_name} ===")
        
        # Получаем агента из state (будет передан через config)
        # Пока используем заглушку - реальный вызов будет в графе
        
        # Агент уже скомпилирован, просто вызываем его
        # Агенты созданы через create_react_agent, поэтому принимают state напрямую
        
        # Заглушка - реальный вызов будет ниже в графе
        pass
    
    return agent_node


# ============================================================================
# ROUTER (CONDITIONAL EDGE)
# ============================================================================

def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Определить продолжить работу или закончить
    
    Args:
        state: Текущее состояние
    
    Returns:
        "continue" - продолжить (есть next_agent)
        "end" - закончить (нет next_agent)
    """
    next_agent = state.get("next_agent")
    
    if next_agent:
        logger.info(f"Router: continue to {next_agent}")
        return "continue"
    else:
        logger.info("Router: end (no next agent)")
        return "end"


def route_to_agent(state: AgentState) -> str:
    """Определить к какому агенту передать управление
    
    Args:
        state: Текущее состояние
    
    Returns:
        Имя агента (например, "pet_memory")
    """
    next_agent = state.get("next_agent")
    
    if not next_agent:
        logger.error("Router called but no next_agent in state!")
        return "supervisor"  # Fallback
    
    logger.info(f"Routing to agent: {next_agent}")
    return next_agent


# ============================================================================
# SUPERVISOR GRAPH
# ============================================================================

class SupervisorGraph:
    """Мульти-агентный граф с supervisor pattern"""
    
    def __init__(
        self,
        agent_factory: AgentFactory,
        user_id: int,
        chat_settings: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            agent_factory: Фабрика агентов
            user_id: ID пользователя
            chat_settings: Настройки чата
        """
        self.agent_factory = agent_factory
        self.user_id = user_id
        self.chat_settings = chat_settings or {}
        
        self.agents = None
        self.handoff_tools = None
        self.graph = None
        
        logger.info(f"SupervisorGraph initialized for user {user_id}")
    
    async def build(self):
        """Построить граф"""
        
        # Создаём агентов
        agents_data = self.agent_factory.create_agents(
            user_id=self.user_id,
            chat_settings=self.chat_settings
        )
        
        self.agents = agents_data["agents"]
        self.handoff_tools = agents_data["handoff_tools"]
        settings = agents_data["settings"]
        
        # Calendar agent
        calendar_agent = await self.agent_factory.create_calendar_agent_if_available(
            user_id=self.user_id,
            chat_settings=self.chat_settings
        )
        
        if calendar_agent:
            self.agents["calendar"] = calendar_agent
            calendar_handoff = self.agent_factory._make_handoff_tool(
                "calendar",
                "Работа с Google Calendar: события, напоминания"
            )
            self.handoff_tools.append(calendar_handoff)
        
        # Создаём LLM для supervisor
        from app.integrations.gigachat_client import create_llm_from_settings
        llm = create_llm_from_settings(self.chat_settings)
        
        # Создаём supervisor node с настройками
        supervisor_node = create_supervisor_node(
            llm,
            self.handoff_tools,
            chat_settings=settings
        )
        
        # === СТРОИМ ГРАФ ===
        
        workflow = StateGraph(AgentState)
        
        # Добавляем supervisor node
        workflow.add_node("supervisor", supervisor_node)
        
        # Добавляем agent nodes
        for agent_name, agent in self.agents.items():
            if agent is not None:  # Пропускаем calendar если его нет
                workflow.add_node(agent_name, agent)
        
        # === EDGES ===
        
        # Start → supervisor
        workflow.set_entry_point("supervisor")
        
        # Supervisor → conditional edge (continue или end)
        def route_from_supervisor(state: AgentState):
            """Роутинг из supervisor"""
            if should_continue(state) == "end":
                return END
            else:
                # Возвращаем имя агента из state
                return route_to_agent(state)
        
        workflow.add_conditional_edges(
            "supervisor",
            route_from_supervisor,  # ✅ Одна функция, которая возвращает либо END либо имя агента
        )
        
        # Каждый агент → обратно к supervisor
        for agent_name in self.agents.keys():
            if self.agents[agent_name] is not None:
                workflow.add_edge(agent_name, "supervisor")
        
        # === COMPILE ===
        
        # Добавляем checkpointer для памяти (опционально)
        checkpointer = MemorySaver()
        
        self.graph = workflow.compile(checkpointer=checkpointer)
        
        logger.info(f"SupervisorGraph compiled with {len(self.agents)} agents")
        
        return self.graph
    
    async def invoke(
        self,
        messages: list,
        user_id: int,
        chat_id: int,
        uploaded_files: list = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """Запустить граф
        
        Args:
            messages: История сообщений
            user_id: ID пользователя
            chat_id: ID чата
            uploaded_files: Загруженные файлы
            config: Конфигурация (для checkpointer)
        
        Returns:
            Финальное состояние
        """
        if not self.graph:
            await self.build()
        
        # Формируем начальное состояние
        initial_state: AgentState = {
            "messages": messages,
            "user_id": user_id,
            "chat_id": chat_id,
            "uploaded_files": uploaded_files or [],
            "generated_files": [],
            "chat_settings": self.chat_settings,
            "current_agent": None,
            "next_agent": None,
        }
        
        # Запускаем граф
        logger.info(f"Starting SupervisorGraph for chat {chat_id}")
        
        final_state = await self.graph.ainvoke(
            initial_state,
            config=config or {"configurable": {"thread_id": str(chat_id)}}
        )
        
        logger.info(f"SupervisorGraph completed for chat {chat_id}")
        
        return final_state


# ============================================================================
# HELPER FUNCTION
# ============================================================================

async def create_supervisor_graph(
    agent_factory: AgentFactory,
    user_id: int,
    chat_settings: Optional[Dict[str, Any]] = None,
):
    """Удобная функция для создания и компиляции графа
    
    Args:
        agent_factory: Фабрика агентов
        user_id: ID пользователя
        chat_settings: Настройки чата
    
    Returns:
        Compiled StateGraph
    """
    supervisor = SupervisorGraph(agent_factory, user_id, chat_settings)
    await supervisor.build()
    return supervisor
