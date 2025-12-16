# app/services/message_service.py

from __future__ import annotations

from typing import List, Optional, Dict, Any
from loguru import logger

from app.repositories.message_repository import MessageRepository
from app.repositories.chat_repository import ChatRepository
from app.dto import MessageCreateDTO, MessageUpdateDTO, MessageResponseDTO, FileMetadataDTO
from app.models.message import Message, MessageRole, MessageType
from app.utils.exceptions import MessageNotFoundException, ChatNotFoundException, AuthorizationException

from app.services.file_service import FileService


def _detect_message_type(files: Optional[List[FileMetadataDTO]]) -> MessageType:
    if not files:
        return MessageType.TEXT
    if len(files) > 1:
        return MessageType.MIXED

    ft = (files[0].file_type or "").lower()
    if ft == "image":
        return MessageType.IMAGE
    if ft == "video":
        return MessageType.VIDEO
    if ft == "audio":
        return MessageType.AUDIO
    return MessageType.DOCUMENT


class MessageService:
    """Сервис для сообщений + привязка файлов + метаданные."""

    def __init__(
        self,
        message_repository: MessageRepository,
        chat_repository: ChatRepository,
        file_service: FileService,
    ):
        self.message_repository = message_repository
        self.chat_repository = chat_repository
        self.file_service = file_service

    async def _assert_chat_access(self, chat_id: int, user_id: int) -> None:
        chat = await self.chat_repository.get_by_id(chat_id)
        if not chat:
            raise ChatNotFoundException(chat_id)
        if chat.user_id != user_id:
            raise AuthorizationException("Нет доступа к этому чату")

    async def _resolve_files(self, user_id: int, file_ids: Optional[List[str]]) -> Optional[List[Dict[str, Any]]]:
        """
        file_ids: список object_name в MinIO
        Возвращает: список dict-ов (как в Message.files JSON), чтобы фронт сразу мог рисовать карточки.
        """
        if not file_ids:
            return None

        resolved: List[Dict[str, Any]] = []
        for fid in file_ids:
            meta = await self.file_service.get_file_metadata(user_id=user_id, file_id=fid)
            resolved.append(meta.model_dump())
        return resolved

    async def create_user_message(
        self,
        chat_id: int,
        user_id: int,
        content: str,
        file_ids: Optional[List[str]] = None,
    ) -> MessageResponseDTO:
        """Создать USER сообщение. Резолвит file_ids -> files metadata."""
        await self._assert_chat_access(chat_id, user_id)

        files_json = await self._resolve_files(user_id=user_id, file_ids=file_ids)
        msg_type = _detect_message_type([FileMetadataDTO(**f) for f in files_json] if files_json else None)

        msg = Message(
            chat_id=chat_id,
            role=MessageRole.USER,
            content=content,
            message_type=msg_type,
            files=files_json,
            metadata_json=None,
            processing_time_ms=None,
        )

        msg = await self.message_repository.create(msg)
        logger.info(f"Created USER message {msg.id} chat={chat_id} type={msg_type}")
        return MessageResponseDTO.model_validate(msg)

    async def create_assistant_message(
        self,
        chat_id: int,
        user_id: int,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[int] = None,
        generated_file_ids: Optional[List[str]] = None,
    ) -> MessageResponseDTO:
        """
        Создать ASSISTANT сообщение.
        generated_file_ids — если ассистент/контент-агент создал файлы и сохранил в MinIO.
        """
        await self._assert_chat_access(chat_id, user_id)

        files_json = await self._resolve_files(user_id=user_id, file_ids=generated_file_ids)
        msg_type = _detect_message_type([FileMetadataDTO(**f) for f in files_json] if files_json else None)

        msg = Message(
            chat_id=chat_id,
            role=MessageRole.ASSISTANT,
            content=content,
            message_type=msg_type if files_json else MessageType.TEXT,
            files=files_json,
            metadata_json=metadata or None,
            processing_time_ms=processing_time_ms,
        )

        msg = await self.message_repository.create(msg)
        logger.info(f"Created ASSISTANT message {msg.id} chat={chat_id}")
        return MessageResponseDTO.model_validate(msg)

    async def add_metadata(
        self,
        message_id: int,
        user_id: int,
        patch: Dict[str, Any],
    ) -> MessageResponseDTO:
        """Добавить/обновить metadata_json (патчем)."""
        msg = await self.message_repository.get_by_id(message_id)
        if not msg:
            raise MessageNotFoundException(message_id)

        # доступ через чат
        await self._assert_chat_access(msg.chat_id, user_id)

        if not msg.metadata_json:
            msg.metadata_json = {}
        msg.metadata_json.update(patch)

        msg = await self.message_repository.update(msg)
        return MessageResponseDTO.model_validate(msg)

    async def list_chat_messages(
        self,
        chat_id: int,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        order: str = "asc",  # "asc" | "desc"
    ) -> List[MessageResponseDTO]:
        """Сообщения для UI (пагинация)."""
        await self._assert_chat_access(chat_id, user_id)

        messages = await self.message_repository.get_chat_messages(chat_id=chat_id, skip=skip, limit=limit)
        # Репозиторий сейчас без order_by — сортируем на уровне сервиса
        messages.sort(key=lambda m: m.created_at, reverse=(order.lower() == "desc"))

        return [MessageResponseDTO.model_validate(m) for m in messages]

    async def get_recent_for_context(
        self,
        chat_id: int,
        user_id: int,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """
        Последние N сообщений для LLM (строго old -> new).
        Учитывает chat.message_limit, если limit не передан.
        """
        chat = await self.chat_repository.get_by_id(chat_id)
        if not chat:
            raise ChatNotFoundException(chat_id)
        if chat.user_id != user_id:
            raise AuthorizationException("Нет доступа к этому чату")

        effective_limit = limit or chat.message_limit or 20
        messages = await self.message_repository.get_chat_messages(chat_id=chat_id, limit=effective_limit)
        messages.sort(key=lambda m: m.created_at)

        return messages

    async def update_user_message(
        self,
        message_id: int,
        user_id: int,
        content: str,
        file_ids: Optional[List[str]] = None,
    ) -> MessageResponseDTO:
        """Редактирование только USER сообщений."""
        msg = await self.message_repository.get_by_id(message_id)
        if not msg:
            raise MessageNotFoundException(message_id)

        await self._assert_chat_access(msg.chat_id, user_id)

        if msg.role != MessageRole.USER:
            raise AuthorizationException("Можно редактировать только свои сообщения")

        msg.content = content

        if file_ids is not None:
            files_json = await self._resolve_files(user_id=user_id, file_ids=file_ids)
            msg.files = files_json
            msg.message_type = _detect_message_type([FileMetadataDTO(**f) for f in files_json] if files_json else None)

        msg = await self.message_repository.update(msg)
        logger.info(f"Updated message {message_id}")
        return MessageResponseDTO.model_validate(msg)

    async def delete_message(
        self,
        message_id: int,
        user_id: int,
    ) -> bool:
        """Soft delete."""
        msg = await self.message_repository.get_by_id(message_id)
        if not msg:
            raise MessageNotFoundException(message_id)

        await self._assert_chat_access(msg.chat_id, user_id)

        msg.soft_delete()
        await self.message_repository.update(msg)
        logger.info(f"Soft deleted message {message_id}")
        return True
