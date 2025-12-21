from __future__ import annotations

from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Состояние мульти-агентной системы"""
    
    # История сообщений (автоматически добавляются новые)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Метаданные запроса
    user_id: int
    chat_id: int
    
    # Загруженные файлы пользователя
    uploaded_files: list[dict]
    
    # Сгенерированные файлы (изображения, графики, аудио, отчёты)
    generated_files: list[dict]
    
    # Настройки чата (из БД)
    chat_settings: Optional[dict]
    
    # Текущий активный агент (для отслеживания)
    current_agent: Optional[str]
    
    # Следующий агент (куда передать управление)
    next_agent: Optional[str]