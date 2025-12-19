from loguru import logger

from app.agents.pet_memory_agent import PetMemoryAgent
from app.agents.document_rag_agent import DocumentRAGAgent
from app.agents.multimodal_agent import MultimodalAgent
from app.agents.web_search_agent import WebSearchAgent
from app.agents.health_nutrition_agent import HealthNutritionAgent
from app.agents.calendar_agent import CalendarAgent
from app.agents.content_generation_agent import ContentGenerationAgent
from app.agents.email_agent import EmailAgent
from app.agents.orchestrator_agent import OrchestratorAgent

from app.services.pet_service import PetService
from app.services.health_record_service import HealthRecordService
from app.services.user_service import UserService
from app.integrations.minio_service import MinioService
from app.integrations.email_service import email_service
from app.rag.rag_service import get_rag_service
from app.integrations.gigachat_client import create_llm_from_settings


class AgentFactory:
    """Фабрика для создания агентов с нужными зависимостями"""

    def __init__(
        self,
        pet_service: PetService,
        health_record_service: HealthRecordService,
        user_service: UserService,
        minio_service: MinioService,
    ):
        self.pet_service = pet_service
        self.health_record_service = health_record_service
        self.user_service = user_service
        self.minio_service = minio_service

        self._llm_factory = create_llm_from_settings
        self.llm = self._llm_factory()

        logger.info("AgentFactory initialized")

    def create_pet_memory_agent(self) -> PetMemoryAgent:
        return PetMemoryAgent(
            pet_service=self.pet_service,
            health_record_service=self.health_record_service,
            llm=self.llm,
        )

    def create_document_rag_agent(self, use_hybrid_retriever: bool = False) -> DocumentRAGAgent:
        return DocumentRAGAgent(
            llm=self.llm,
            use_hybrid_retriever=use_hybrid_retriever,
        )

    def create_multimodal_agent(self) -> MultimodalAgent:
        return MultimodalAgent(
            minio_service=self.minio_service,
            llm=self.llm,
        )

    def create_web_search_agent(self) -> WebSearchAgent:
        return WebSearchAgent(llm=self.llm)

    def create_health_nutrition_agent(self) -> HealthNutritionAgent:
        return HealthNutritionAgent(
            pet_service=self.pet_service,
            health_record_service=self.health_record_service,
            llm=self.llm,
        )

    def create_calendar_agent(self) -> CalendarAgent:
        return CalendarAgent(
            user_service=self.user_service,
            llm=self.llm,
        )

    def create_content_generation_agent(self) -> ContentGenerationAgent:
        return ContentGenerationAgent(
            minio=self.minio_service,
            llm=self.llm,
        )

    def create_email_agent(self) -> EmailAgent:
        return EmailAgent(
            email_service=email_service,
            llm=self.llm,
        )

    def create_orchestrator(self) -> OrchestratorAgent:
        logger.info("Creating all agents for Orchestrator...")

        pet_memory_agent = self.create_pet_memory_agent()
        document_rag_agent = self.create_document_rag_agent()
        multimodal_agent = self.create_multimodal_agent()
        web_search_agent = self.create_web_search_agent()
        health_nutrition_agent = self.create_health_nutrition_agent()
        calendar_agent = self.create_calendar_agent()
        content_generation_agent = self.create_content_generation_agent()
        email_agent = self.create_email_agent()

        orchestrator = OrchestratorAgent(
            pet_memory_agent=pet_memory_agent,
            document_rag_agent=document_rag_agent,
            multimodal_agent=multimodal_agent,
            web_search_agent=web_search_agent,
            health_nutrition_agent=health_nutrition_agent,
            calendar_agent=calendar_agent,
            content_generation_agent=content_generation_agent,
            email_agent=email_agent,
            llm=self.llm,
            llm_factory=self._llm_factory,
        )

        logger.info("Orchestrator created with 8 specialized agents")
        return orchestrator


_agent_factory: AgentFactory = None


def get_agent_factory(
    pet_service: PetService,
    health_record_service: HealthRecordService,
    user_service: UserService,
    minio_service: MinioService,
) -> AgentFactory:
    global _agent_factory

    if _agent_factory is None:
        _agent_factory = AgentFactory(
            pet_service=pet_service,
            health_record_service=health_record_service,
            user_service=user_service,
            minio_service=minio_service,
        )

    return _agent_factory
