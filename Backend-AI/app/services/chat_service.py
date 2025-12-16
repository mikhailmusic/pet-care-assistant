from __future__ import annotations

from typing import Any, Dict, Optional, List
from loguru import logger
from dataclasses import dataclass, field

import time
from app.services.message_service import MessageService
from app.repositories.chat_repository import ChatRepository
from app.dto import ChatCreateDTO, ChatUpdateDTO, ChatResponseDTO, ChatListItemDTO
from app.dto import ChatSettingsDTO, MessageCreateDTO, MessageResponseDTO
from app.utils.exceptions import ChatNotFoundException, AuthorizationException

@dataclass
class OrchestratorResult:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_files: List[Dict[str, Any]] = field(default_factory=list)

class ChatService:
    def __init__(
        self,
        chat_repository: ChatRepository,
        message_service: MessageService,
        orchestrator,  # твой Orchestrator (интерфейс ниже)
    ):
        self.repo = chat_repository
        self.message_service = message_service
        self.orchestrator = orchestrator

    async def send_message(
        self,
        chat_id: int,
        user_id: int,
        dto: MessageCreateDTO,
    ) -> MessageResponseDTO:
        """
        Полный цикл:
        1) сохранить user message
        2) собрать history + settings
        3) прогнать orchestrator
        4) сохранить assistant message (с metadata, processing_time)
        5) вернуть assistant message (DTO)
        """
        start = time.monotonic()

        # 0) (опционально) быстрая проверка доступа — можно не делать,
        # т.к. message_service.create_user_message уже проверяет чат/права.
        chat = await self.repo.get_by_id(chat_id)
        if not chat:
            raise ChatNotFoundException(chat_id)
        if chat.user_id != user_id:
            raise AuthorizationException("Нет доступа к этому чату")

        # 1) создаём сообщение пользователя
        user_msg = await self.message_service.create_user_message(
            chat_id=chat_id,
            user_id=user_id,
            message_dto=dto,
        )

        # 2) история для контекста (объекты Message, не DTO)
        history = await self.message_service.get_recent_messages_for_context(
            chat_id=chat_id,
            user_id=user_id,
            limit=None,  # пусть лимит возьмётся из chat.message_limit внутри MessageService
        )

        # 3) настройки чата для оркестратора
        settings: ChatSettingsDTO = await self.get_chat_settings(chat_id=chat_id, user_id=user_id)

        # uploaded_files: берём из dto.files (или из metadata user message)
        uploaded_files = dto.files or []
        # если ты хранишь в metadata_json["files"] — можно так:
        # uploaded_files = (user_msg.metadata_json or {}).get("files", []) or []

        # 4) оркестратор
        # Ожидаемый интерфейс:
        # result = await orchestrator.run(messages=history, chat_settings=settings, uploaded_files=uploaded_files)
        result: OrchestratorResult = await self.orchestrator.run(
            messages=history,
            chat_settings=settings,
            uploaded_files=uploaded_files,
            chat_id=chat_id,
            user_id=user_id,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # 5) metadata ассистента: сюда удобно положить generated_files и любые флаги
        assistant_metadata: Dict[str, Any] = dict(result.metadata or {})
        if result.generated_files:
            assistant_metadata["generated_content"] = result.generated_files

        # 6) создаём сообщение ассистента
        assistant_msg = await self.message_service.create_assistant_message(
            chat_id=chat_id,
            user_id=user_id,
            content=result.text,
            metadata=assistant_metadata or None,
            processing_time_ms=elapsed_ms,
        )

        logger.info(f"send_message chat={chat_id} user={user_id} took={elapsed_ms}ms")
        return assistant_msg


    async def create_chat(self, user_id: int, dto: ChatCreateDTO) -> ChatResponseDTO:
        from app.models.chat import Chat

        chat = Chat(
            user_id=user_id,
            title=dto.title,
            description=dto.description,
        )

        chat = await self.repo.create(chat)
        logger.info(f"Created chat {chat.id} for user {user_id}")

        resp = ChatResponseDTO.model_validate(chat)
        resp.message_count = 0
        return resp

    async def get_chat(self, chat_id: int, user_id: int, with_messages: bool = False) -> ChatResponseDTO:
        if with_messages:
            chat = await self.repo.get_with_messages(chat_id)
        else:
            chat = await self.repo.get_by_id(chat_id)

        if not chat:
            raise ChatNotFoundException(chat_id)
        if chat.user_id != user_id:
            raise AuthorizationException("Нет доступа к этому чату")

        resp = ChatResponseDTO.model_validate(chat)

        # message_count — если загрузили сообщения, считаем реально
        if with_messages and getattr(chat, "messages", None) is not None:
            resp.message_count = len([m for m in chat.messages if not m.is_deleted])
        else:
            resp.message_count = 0  # можно заменить на отдельный COUNT-запрос при желании

        return resp

    async def list_user_chats(self, user_id: int, skip: int = 0, limit: int = 100) -> List[ChatListItemDTO]:
        rows = await self.repo.get_list_items_with_stats(user_id=user_id, skip=skip, limit=limit)

        items: List[ChatListItemDTO] = []
        for chat, message_count, last_message_at in rows:
            dto = ChatListItemDTO.model_validate(chat)
            dto.message_count = int(message_count or 0)
            dto.last_message_at = last_message_at
            items.append(dto)

        logger.info(f"Listed {len(items)} chats for user {user_id} (skip={skip}, limit={limit})")
        return items

    async def update_chat(self, chat_id: int, user_id: int, dto: ChatUpdateDTO) -> ChatResponseDTO:
        chat = await self.repo.get_by_id(chat_id)
        if not chat:
            raise ChatNotFoundException(chat_id)
        if chat.user_id != user_id:
            raise AuthorizationException("Нет доступа к этому чату")

        update_data = dto.model_dump(exclude_unset=True)
        for k, v in update_data.items():
            setattr(chat, k, v)

        chat = await self.repo.update(chat)
        logger.info(f"Updated chat {chat_id}: {list(update_data.keys())}")

        resp = ChatResponseDTO.model_validate(chat)
        resp.message_count = 0
        return resp

    async def delete_chat(self, chat_id: int, user_id: int) -> bool:
        chat = await self.repo.get_by_id(chat_id)
        if not chat:
            raise ChatNotFoundException(chat_id)
        if chat.user_id != user_id:
            raise AuthorizationException("Нет доступа к этому чату")

        chat.soft_delete()
        await self.repo.update(chat)

        logger.info(f"Soft deleted chat {chat_id}")
        return True

    async def get_chat_settings(self, chat_id: int, user_id: int) -> ChatSettingsDTO:
        chat = await self.repo.get_by_id(chat_id)
        if not chat:
            raise ChatNotFoundException(chat_id)
        if chat.user_id != user_id:
            raise AuthorizationException("Нет доступа к этому чату")

        return ChatSettingsDTO(
            web_search_enabled=chat.web_search_enabled,
            message_limit=chat.message_limit,
            temperature=chat.temperature,
            gigachat_model=chat.gigachat_model,
            image_generation_enabled=chat.image_generation_enabled,
            voice_response_enabled=chat.voice_response_enabled,
            max_tokens=chat.max_tokens,
        )

    async def update_message_and_regenerate(
        self,
        message_id: int,
        user_id: int,
        content: str,
        file_ids: Optional[List[str]] = None,
    ) -> MessageResponseDTO:
        """
        Обновить USER сообщение и автоматически перегенерировать ответ ассистента.

        Полный цикл:
        1) Обновить USER сообщение (content, files)
        2) Удалить все последующие сообщения (отбросить историю)
        3) Собрать контекст и настройки
        4) Вызвать оркестратор
        5) Сохранить новый ASSISTANT ответ
        6) Вернуть ASSISTANT ответ

        Это аналог send_message, но вместо создания нового USER сообщения - обновляем существующее.
        """
        start = time.monotonic()

        # 1) Обновляем USER сообщение + удаляем последующие
        _, deleted_count = await self.message_service.update_user_message(
            message_id=message_id,
            user_id=user_id,
            content=content,
            file_ids=file_ids,
            delete_subsequent=True,  # всегда удаляем последующие
        )

        logger.info(f"Updated message {message_id}, deleted {deleted_count} subsequent messages")

        # Получаем chat_id из обновленного сообщения
        msg = await self.message_service.message_repository.get_by_id(message_id)
        chat_id = msg.chat_id

        # 2) История для контекста
        history = await self.message_service.get_recent_messages_for_context(
            chat_id=chat_id,
            user_id=user_id,
            limit=None,
        )

        # 3) Настройки чата
        settings: ChatSettingsDTO = await self.get_chat_settings(chat_id=chat_id, user_id=user_id)

        # 4) Файлы из обновленного сообщения
        uploaded_files = file_ids or []

        # 5) Оркестратор
        result: OrchestratorResult = await self.orchestrator.run(
            messages=history,
            chat_settings=settings,
            uploaded_files=uploaded_files,
            chat_id=chat_id,
            user_id=user_id,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # 6) Metadata ассистента
        assistant_metadata: Dict[str, Any] = dict(result.metadata or {})
        if result.generated_files:
            assistant_metadata["generated_content"] = result.generated_files

        # 7) Создаем новое сообщение ассистента
        assistant_msg = await self.message_service.create_assistant_message(
            chat_id=chat_id,
            user_id=user_id,
            content=result.text,
            metadata=assistant_metadata or None,
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            f"update_message_and_regenerate: message={message_id} chat={chat_id} "
            f"deleted={deleted_count} took={elapsed_ms}ms"
        )

        return assistant_msg
