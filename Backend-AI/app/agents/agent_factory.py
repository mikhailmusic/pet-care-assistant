from __future__ import annotations

from typing import Optional, Dict, Any
from loguru import logger

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from app.services.pet_service import PetService
from app.services.health_record_service import HealthRecordService
from app.services.user_service import UserService
from app.integrations.email_service import EmailService
from app.integrations.minio_service import MinioService, minio_service
from app.integrations.ddg_client import duckduckgo_service
from app.integrations.google_calendar_client import GoogleCalendarClient
from app.integrations.gigachat_client import create_llm_from_settings
from app.rag.rag_service import get_rag_service
from app.config import settings

from app.agents.pet_memory_agent import create_pet_memory_agent
from app.agents.web_search_agent import create_web_search_agent
from app.agents.email_agent import create_email_agent
from app.agents.document_rag_agent import create_document_rag_agent
from app.agents.content_generation_agent import create_content_generation_agent
from app.agents.multimodal_agent import create_multimodal_agent
from app.agents.health_nutrition_agent import create_health_nutrition_agent
from app.agents.calendar_agent import create_calendar_agent


class AgentFactory:
    """Фабрика для создания всех агентов и supervisor"""
    
    def __init__(
        self,
        pet_service: PetService,
        health_record_service: HealthRecordService,
        email_service: EmailService,
        user_service: UserService,  # Для Google Calendar credentials
    ):
        """
        Args:
            pet_service: Сервис для работы с питомцами
            health_record_service: Сервис для медицинских записей
            email_service: Сервис для отправки email
            user_service: Сервис пользователей (для Google credentials)
        """
        self.pet_service = pet_service
        self.health_record_service = health_record_service
        self.email_service = email_service
        self.user_service = user_service
        
        logger.info("AgentFactory initialized")
    
    def create_agents(
        self,
        user_id: int,
        chat_settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Создать агенты с учётом настроек чата"""
        
        llm = create_llm_from_settings(chat_settings)
        
        settings = chat_settings or {}
        web_search_enabled = settings.get("web_search_enabled", True)
        image_generation_enabled = settings.get("image_generation_enabled", True)
        voice_response_enabled = settings.get("voice_response_enabled", True)
        
        # === СОЗДАЁМ АГЕНТОВ ===
        
        agents = {}
        
        # 1. PetMemoryAgent (всегда доступен)
        agents["pet_memory"] = create_pet_memory_agent(
            pet_service=self.pet_service,
            health_service=self.health_record_service,
            llm=llm,
            name="pet_memory"
        )
        
        # 2. WebSearchAgent (только если web_search_enabled)
        if web_search_enabled:
            agents["web_search"] = create_web_search_agent(
                duckduckgo_service=duckduckgo_service,
                llm=llm,
                name="web_search"
            )
            logger.info("WebSearchAgent enabled")
        else:
            agents["web_search"] = None
            logger.info("WebSearchAgent disabled by chat settings")
        
        # 3. EmailAgent (всегда доступен)
        agents["email"] = create_email_agent(
            email_service=self.email_service,
            llm=llm,
            name="email"
        )
        
        # 4. DocumentRAGAgent (всегда доступен)
        rag_service = get_rag_service(use_hybrid_retriever=False)
        agents["document_rag"] = create_document_rag_agent(
            rag_service=rag_service,
            llm=llm,
            name="document_rag"
        )
        
        # 5. ContentGenerationAgent (только если image_generation_enabled)
        if image_generation_enabled:
            agents["content_generation"] = create_content_generation_agent(
                minio_service=minio_service,
                llm=llm,
                name="content_generation"
            )
            logger.info("ContentGenerationAgent enabled")
        else:
            agents["content_generation"] = None
            logger.info("ContentGenerationAgent disabled by chat settings")
        
        # 6. MultimodalAgent (всегда доступен - анализ загруженных файлов)
        agents["multimodal"] = create_multimodal_agent(
            minio_service=minio_service,
            llm=llm,
            name="multimodal"
        )
        
        # 7. HealthNutritionAgent (всегда доступен)
        agents["health_nutrition"] = create_health_nutrition_agent(
            pet_service=self.pet_service,
            health_service=self.health_record_service,
            llm=llm,
            name="health_nutrition"
        )
        
        # 8. CalendarAgent (создаётся асинхронно позже)
        agents["calendar"] = None
        
        # === СОЗДАЁМ HANDOFF TOOLS (только для доступных агентов) ===
        
        handoff_tools = self._create_handoff_tools(agents)
        
        logger.info(
            f"Created agents: {len([a for a in agents.values() if a is not None])} enabled, "
            f"{len([a for a in agents.values() if a is None])} disabled"
        )
        
        return {
            "agents": agents,
            "handoff_tools": handoff_tools,
            "settings": {
                "web_search_enabled": web_search_enabled,
                "image_generation_enabled": image_generation_enabled,
                "voice_response_enabled": voice_response_enabled,
            }
        }
    
    async def create_calendar_agent_if_available(
        self,
        user_id: int,
        chat_settings: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Создать CalendarAgent если у пользователя есть Google credentials
        
        Args:
            user_id: ID пользователя
            chat_settings: Настройки чата
        
        Returns:
            Compiled CalendarAgent или None
        """
        try:
            # Получаем credentials
            creds_json = await self.user_service.get_google_credentials(user_id)
            
            if not creds_json:
                logger.info(f"No Google Calendar credentials for user {user_id}")
                return None
            
            # Инициализируем клиент
            calendar_client = GoogleCalendarClient()
            
            try:
                calendar_client.set_credentials_from_json(creds_json)
            except Exception as e:
                logger.error(f"Invalid Google credentials for user {user_id}: {e}")
                return None
            
            # Получаем timezone (из настроек или default)
            user_timezone = settings.DEFAULT_TIMEZONE
            if chat_settings and "user_timezone" in chat_settings:
                user_timezone = chat_settings["user_timezone"]
            
            # Создаём LLM
            llm = create_llm_from_settings(chat_settings)
            
            # Создаём агента
            calendar_agent = create_calendar_agent(
                calendar_client=calendar_client,
                user_timezone=user_timezone,
                llm=llm,
                name="calendar"
            )
            
            logger.info(f"Created CalendarAgent for user {user_id}")
            return calendar_agent
            
        except Exception as e:
            logger.error(f"Failed to create CalendarAgent for user {user_id}: {e}")
            return None
    
    def _create_handoff_tools(self, agents: Dict[str, Any]) -> list:
        """Создать handoff tools только для доступных агентов"""
        
        handoff_tools = []
        
        agent_descriptions = {
            "pet_memory": "Управление данными о питомцах и медицинских записях",
            "web_search": "Поиск актуальной информации в интернете",
            "email": "Отправка email писем",
            "document_rag": "Работа с документами: индексация и поиск",
            "content_generation": "Генерация контента: изображения, графики, аудио, отчёты",
            "multimodal": "Анализ мультимедиа: изображения, аудио, видео",
            "health_nutrition": "Анализ здоровья и питания питомцев",
            "calendar": "Работа с Google Calendar: события, напоминания",
        }
        
        for agent_name, description in agent_descriptions.items():
            # Пропускаем агентов которых нет (None)
            if agents.get(agent_name) is None:
                logger.debug(f"Skipping handoff tool for disabled agent: {agent_name}")
                continue
            
            # Создаём handoff tool
            handoff_tool = self._make_handoff_tool(agent_name, description)
            handoff_tools.append(handoff_tool)
        
        logger.info(f"Created {len(handoff_tools)} handoff tools")
        return handoff_tools
    
    def _make_handoff_tool(self, agent_name: str, description: str):
        """Создать handoff tool для конкретного агента
        
        Args:
            agent_name: Имя агента (например, "pet_memory")
            description: Описание функционала агента
        
        Returns:
            Tool функция
        """
        
        # 1. Создаём docstring с f-string
        tool_description = f"""Передать управление агенту {agent_name}.

    {description}

    Args:
        reason: Причина передачи управления (опционально)

    Returns:
        Подтверждение передачи
    """
        
        # 2. Создаём функцию с правильным docstring
        def transfer_tool(reason: str = "") -> str:
            logger.info(f"Handoff to {agent_name}: {reason}")
            return f"Transferred to {agent_name}"
        
        # 3. Устанавливаем docstring
        transfer_tool.__doc__ = tool_description
        
        # 4. Устанавливаем имя функции (для tool декоратора)
        transfer_tool.__name__ = f"transfer_to_{agent_name}"
        
        # 5. Применяем декоратор tool БЕЗ параметра name
        handoff_tool = tool(transfer_tool)
        
        return handoff_tool


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Будет инициализирован в ChatService
_agent_factory: Optional[AgentFactory] = None


def get_agent_factory(
    pet_service: PetService = None,
    health_record_service: HealthRecordService = None,
    email_service: EmailService = None,
    user_service = None,
) -> AgentFactory:
    """Получить singleton AgentFactory"""
    global _agent_factory
    
    if _agent_factory is None:
        if not all([pet_service, health_record_service, email_service, user_service]):
            raise RuntimeError("AgentFactory not initialized. Provide all services.")
        
        _agent_factory = AgentFactory(
            pet_service=pet_service,
            health_record_service=health_record_service,
            email_service=email_service,
            user_service=user_service,
        )
    
    return _agent_factory