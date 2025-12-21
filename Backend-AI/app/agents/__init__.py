# app/agents/__init__.py

from app.agents.agent_factory import AgentFactory
from app.agents.supervisor import create_supervisor_graph, SupervisorGraph
from app.agents.state import AgentState

# Фабричные функции для создания агентов (опционально экспортируем)
from app.agents.pet_memory_agent import create_pet_memory_agent
from app.agents.web_search_agent import create_web_search_agent
from app.agents.email_agent import create_email_agent
from app.agents.document_rag_agent import create_document_rag_agent
from app.agents.content_generation_agent import create_content_generation_agent
from app.agents.multimodal_agent import create_multimodal_agent
from app.agents.health_nutrition_agent import create_health_nutrition_agent
from app.agents.calendar_agent import create_calendar_agent

__all__ = [
    # Основные классы
    "AgentFactory",
    "SupervisorGraph",
    "create_supervisor_graph",
    "AgentState",
    
    # Фабричные функции (если нужны снаружи)
    "create_pet_memory_agent",
    "create_web_search_agent",
    "create_email_agent",
    "create_document_rag_agent",
    "create_content_generation_agent",
    "create_multimodal_agent",
    "create_health_nutrition_agent",
    "create_calendar_agent",
]