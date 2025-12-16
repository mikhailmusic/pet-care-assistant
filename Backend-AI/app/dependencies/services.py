from typing import Annotated

from fastapi import Depends

from app.services.user_service import UserService
from app.services.chat_service import ChatService
from app.services.pet_service import PetService
from app.integrations.minio_service import minio_service
from app.services.file_service import FileService
from app.services.message_service import MessageService

from app.repositories.user_repository import UserRepository
from app.repositories.chat_repository import ChatRepository
from app.repositories.pet_repository import PetRepository
from app.repositories.message_repository import MessageRepository

from .repositories import (
    get_user_repository,
    get_chat_repository,
    get_pet_repository,
    get_message_repository
)


UserRepositoryDep = Annotated[UserRepository, Depends(get_user_repository)]
ChatRepositoryDep = Annotated[ChatRepository, Depends(get_chat_repository)]
PetRepositoryDep = Annotated[PetRepository, Depends(get_pet_repository)]
MessageRepositoryDep = Annotated[MessageRepository, Depends(get_message_repository)]


def get_user_service(user_repo: UserRepositoryDep) -> UserService:
    return UserService(user_repo)


def get_chat_service(chat_repo: ChatRepositoryDep) -> ChatService:
    return ChatService(chat_repo)


def get_pet_service(pet_repo: PetRepositoryDep) -> PetService:
    return PetService(pet_repo)


def get_file_service() -> FileService:
    return FileService(minio_service=minio_service)

async def get_message_service(msg_repo: MessageRepositoryDep, chat_repo: ChatRepositoryDep) -> MessageService:
    file_service = get_file_service()
    return MessageService(
        message_repository=msg_repo,
        chat_repository=chat_repo,
        file_service=file_service,
    )

UserServiceDep = Annotated[UserService, Depends(get_user_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
PetServiceDep = Annotated[PetService, Depends(get_pet_service)]
FileServiceDep = Annotated[FileService, Depends(get_file_service)]
MessageServiceDep = Annotated[MessageService, Depends(get_message_service)]