from typing import Annotated

from fastapi import Depends

from app.services.user_service import UserService
from app.services.chat_service import ChatService
from app.services.pet_service import PetService
from app.services.health_record_service import HealthRecordService
from app.integrations.email_service import EmailService
from app.integrations.minio_service import minio_service
from app.services.file_service import FileService
from app.services.message_service import MessageService

from app.repositories.user_repository import UserRepository
from app.repositories.chat_repository import ChatRepository
from app.repositories.pet_repository import PetRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.health_record_repository import HealthRecordRepository

from app.agents.agent_factory import AgentFactory

from .repositories import (
    get_user_repository,
    get_chat_repository,
    get_pet_repository,
    get_message_repository,
    get_health_record_repository,
)


UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]
ChatRepositoryDep = Annotated[ChatRepository, Depends(get_chat_repository)]
PetRepositoryDep = Annotated[PetRepository, Depends(get_pet_repository)]
MessageRepositoryDep = Annotated[MessageRepository, Depends(get_message_repository)]
HealthRecordRepositoryDep = Annotated[HealthRecordRepository, Depends(get_health_record_repository)]


def get_user_service(user_repo: UserRepositoryDep) -> UserService:
    return UserService(user_repo)


def get_pet_service(pet_repo: PetRepositoryDep) -> PetService:
    return PetService(pet_repo)


def get_health_record_service(
    health_repo: HealthRecordRepositoryDep,
    pet_repo: PetRepositoryDep,
) -> HealthRecordService:
    return HealthRecordService(
        health_repo=health_repo,
        pet_repository=pet_repo,
    )


def get_email_service() -> EmailService:
    """Создать EmailService"""
    from app.integrations.email_service import EmailService
    return EmailService()


def get_file_service() -> FileService:
    return FileService(minio_service=minio_service)


def get_message_service(
    msg_repo: MessageRepositoryDep,
    chat_repo: ChatRepositoryDep
) -> MessageService:
    file_service = get_file_service()
    return MessageService(
        message_repository=msg_repo,
        chat_repository=chat_repo,
        file_service=file_service,
    )


def get_chat_service(
    chat_repo: ChatRepositoryDep,
    msg_repo: MessageRepositoryDep,
    pet_repo: PetRepositoryDep,
    health_repo: HealthRecordRepositoryDep,
    user_repo: UserRepositoryDep,
) -> ChatService:
    """Создать ChatService с AgentFactory"""

    # Создаём сервисы
    message_service = MessageService(
        message_repository=msg_repo,
        chat_repository=chat_repo,
        file_service=get_file_service(),
    )

    pet_service = PetService(pet_repo)
    
    health_record_service = HealthRecordService(
        health_repo=health_repo,
        pet_repository=pet_repo,
    )
    
    user_service = UserService(user_repo)
    
    email_service = get_email_service()

    agent_factory = AgentFactory(
        pet_service=pet_service,
        health_record_service=health_record_service,
        email_service=email_service,
        user_service=user_service,
    )

    return ChatService(
        chat_repository=chat_repo,
        message_service=message_service,
        agent_factory=agent_factory,
    )


UserServiceDep = Annotated[UserService, Depends(get_user_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
PetServiceDep = Annotated[PetService, Depends(get_pet_service)]
HealthRecordServiceDep = Annotated[HealthRecordService, Depends(get_health_record_service)]
FileServiceDep = Annotated[FileService, Depends(get_file_service)]
MessageServiceDep = Annotated[MessageService, Depends(get_message_service)]
EmailServiceDep = Annotated[EmailService, Depends(get_email_service)]