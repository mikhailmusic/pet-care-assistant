from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations import get_db
from app.repositories.user_repository import UserRepository
from app.repositories.chat_repository import ChatRepository
from app.repositories.pet_repository import PetRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.health_record_repository import HealthRecordRepository


DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_user_repository(db: DbSession) -> UserRepository:
    return UserRepository(db)


def get_chat_repository(db: DbSession) -> ChatRepository:
    return ChatRepository(db)


def get_pet_repository(db: DbSession) -> PetRepository:
    return PetRepository(db)


def get_message_repository(db: DbSession) -> MessageRepository:
    return MessageRepository(db)


def get_health_record_repository(db: DbSession) -> HealthRecordRepository:
    return HealthRecordRepository(db)