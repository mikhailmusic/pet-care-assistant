from __future__ import annotations

from typing import Any, Dict, Optional, List
from loguru import logger
import time

from langchain_core.messages import HumanMessage, AIMessage

from app.services.message_service import MessageService
from app.repositories.chat_repository import ChatRepository
from app.dto import ChatCreateDTO, ChatUpdateDTO, ChatResponseDTO, ChatListItemDTO
from app.dto import ChatSettingsDTO, MessageCreateDTO, MessageResponseDTO
from app.utils.exceptions import ChatNotFoundException, AuthorizationException

from app.agents.agent_factory import AgentFactory
from app.agents.supervisor import create_supervisor_graph


class ChatService:
    def __init__(
        self,
        chat_repository: ChatRepository,
        message_service: MessageService,
        agent_factory: AgentFactory,
    ):
        self.repo = chat_repository
        self.message_service = message_service
        self.agent_factory = agent_factory

    async def send_message(
        self,
        chat_id: int,
        user_id: int,
        dto: MessageCreateDTO,
    ) -> MessageResponseDTO:
        
        start = time.monotonic()
       
        chat = await self.repo.get_by_id(chat_id)
        if not chat:
            raise ChatNotFoundException(chat_id)
        if chat.user_id != user_id:
            raise AuthorizationException("Нет доступа к этому чату")

        # === 2. СОЗДАЁМ USER MESSAGE ===
        
        user_msg = await self.message_service.create_user_message(
            chat_id=chat_id,
            user_id=user_id,
            message_dto=dto,
        )

        # === 3. ПОЛУЧАЕМ КОНТЕКСТ ===
        
        # История сообщений
        history = await self.message_service.get_recent_messages_for_context(
            chat_id=chat_id,
            user_id=user_id,
            limit=None,
        )
        
        # Настройки чата
        settings: ChatSettingsDTO = await self.get_chat_settings(
            chat_id=chat_id,
            user_id=user_id
        )
        
        # Загруженные файлы
        uploaded_files = user_msg.files or []

        # === 4. КОНВЕРТИРУЕМ ИСТОРИЮ В LANGCHAIN MESSAGES ===
        
        langchain_messages = self._convert_to_langchain_messages(history)

        # === 5. СОЗДАЁМ И ЗАПУСКАЕМ SUPERVISOR ===
        
        # Формируем chat_settings для LLM
        chat_settings_dict = {
            "gigachat_model": settings.gigachat_model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "web_search_enabled": settings.web_search_enabled,
            "image_generation_enabled": settings.image_generation_enabled,
            "voice_response_enabled": settings.voice_response_enabled,
        }
        
        # Создаём supervisor graph
        supervisor = await create_supervisor_graph(
            agent_factory=self.agent_factory,
            user_id=user_id,
            chat_settings=chat_settings_dict,
        )
        
        # Запускаем граф
        final_state = await supervisor.invoke(
            messages=langchain_messages,
            user_id=user_id,
            chat_id=chat_id,
            uploaded_files=uploaded_files,
            config={"configurable": {"thread_id": str(chat_id)}}
        )

        # === 6. ИЗВЛЕКАЕМ РЕЗУЛЬТАТ ===
        
        # Последнее сообщение от ассистента
        final_messages = final_state["messages"]
        last_ai_message = None
        
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            logger.error(f"No AI message in final state for chat {chat_id}")
            assistant_content = "Извините, произошла ошибка при обработке запроса."
        else:
            assistant_content = last_ai_message.content or "Обработано."

        # Извлекаем generated_files из state
        generated_files = final_state.get("generated_files", [])

        # === 7. СОХРАНЯЕМ GOOGLE CALENDAR CREDENTIALS (если обновились) ===
        
        # Проверяем есть ли calendar agent в supervisor
        if supervisor.agents.get("calendar"):
            try:
                calendar_client = supervisor.agents["calendar"]._calendar_client
                new_creds_json = calendar_client.get_credentials_json()
                
                # Сохраняем если изменились
                await self.agent_factory.user_service.add_google_credentials(
                    user_id, new_creds_json
                )
                logger.info(f"Updated Google credentials for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to save Google credentials: {e}")

        # === 8. ФОРМИРУЕМ METADATA ===
        
        assistant_metadata: Dict[str, Any] = {}
        
        if generated_files:
            assistant_metadata["generated_content"] = generated_files

        # === 9. СОЗДАЁМ ASSISTANT MESSAGE ===
        
        elapsed_ms = int((time.monotonic() - start) * 1000)
        
        assistant_msg = await self.message_service.create_assistant_message(
            chat_id=chat_id,
            user_id=user_id,
            content=assistant_content,
            metadata=assistant_metadata or None,
            processing_time_ms=elapsed_ms,
        )

        logger.info(f"send_message chat={chat_id} user={user_id} took={elapsed_ms}ms")
        return assistant_msg

    async def update_message_and_regenerate(
        self,
        message_id: int,
        user_id: int,
        content: str,
        file_ids: Optional[List[str]] = None,
    ) -> MessageResponseDTO:
        """Обновить USER сообщение и перегенерировать ответ"""
        
        start = time.monotonic()

        # === 1. ОБНОВЛЯЕМ USER MESSAGE + УДАЛЯЕМ ПОСЛЕДУЮЩИЕ ===
        
        _, deleted_count = await self.message_service.update_user_message(
            message_id=message_id,
            user_id=user_id,
            content=content,
            file_ids=file_ids,
            delete_subsequent=True,
        )

        logger.info(f"Updated message {message_id}, deleted {deleted_count} subsequent messages")

        # === 2. ПОЛУЧАЕМ CHAT_ID ===
        
        msg = await self.message_service.message_repository.get_by_id(message_id)
        chat_id = msg.chat_id

        # === 3. ПОЛУЧАЕМ КОНТЕКСТ ===
        
        history = await self.message_service.get_recent_messages_for_context(
            chat_id=chat_id,
            user_id=user_id,
            limit=None,
        )
        
        settings: ChatSettingsDTO = await self.get_chat_settings(
            chat_id=chat_id,
            user_id=user_id
        )
        
        uploaded_files = msg.files or []

        # === 4. КОНВЕРТИРУЕМ ИСТОРИЮ ===
        
        langchain_messages = self._convert_to_langchain_messages(history)

        # === 5. СОЗДАЁМ И ЗАПУСКАЕМ SUPERVISOR ===
        
        chat_settings_dict = {
            "gigachat_model": settings.gigachat_model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "web_search_enabled": settings.web_search_enabled,
            "image_generation_enabled": settings.image_generation_enabled,
            "voice_response_enabled": settings.voice_response_enabled,
        }
        
        supervisor = await create_supervisor_graph(
            agent_factory=self.agent_factory,
            user_id=user_id,
            chat_settings=chat_settings_dict,
        )
        
        final_state = await supervisor.invoke(
            messages=langchain_messages,
            user_id=user_id,
            chat_id=chat_id,
            uploaded_files=uploaded_files,
            config={"configurable": {"thread_id": str(chat_id)}}
        )

        # === 6. ИЗВЛЕКАЕМ РЕЗУЛЬТАТ ===
        
        final_messages = final_state["messages"]
        last_ai_message = None
        
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            assistant_content = "Извините, произошла ошибка при обработке запроса."
        else:
            assistant_content = last_ai_message.content or "Обработано."

        generated_files = final_state.get("generated_files", [])

        # === 7. СОХРАНЯЕМ GOOGLE CREDENTIALS ===
        
        if supervisor.agents.get("calendar"):
            try:
                calendar_client = supervisor.agents["calendar"]._calendar_client
                new_creds_json = calendar_client.get_credentials_json()
                await self.agent_factory.user_service.add_google_credentials(
                    user_id, new_creds_json
                )
            except Exception as e:
                logger.warning(f"Failed to save Google credentials: {e}")

        # === 8. METADATA ===
        
        assistant_metadata: Dict[str, Any] = {}
        if generated_files:
            assistant_metadata["generated_content"] = generated_files

        # === 9. СОЗДАЁМ НОВОЕ ASSISTANT MESSAGE ===
        
        elapsed_ms = int((time.monotonic() - start) * 1000)
        
        assistant_msg = await self.message_service.create_assistant_message(
            chat_id=chat_id,
            user_id=user_id,
            content=assistant_content,
            metadata=assistant_metadata or None,
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            f"update_message_and_regenerate: message={message_id} chat={chat_id} "
            f"deleted={deleted_count} took={elapsed_ms}ms"
        )

        return assistant_msg

    def _convert_to_langchain_messages(self, history: List) -> List:
        from app.models.message import MessageRole
        
        langchain_messages = []
        
        for msg in history:
            content = msg.content or ""
            
            if msg.role == MessageRole.USER:
                langchain_messages.append(HumanMessage(content=content))
            elif msg.role == MessageRole.ASSISTANT:
                langchain_messages.append(AIMessage(content=content))
            # Пропускаем SYSTEM или другие роли
        
        return langchain_messages


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

        if with_messages and getattr(chat, "messages", None) is not None:
            resp.message_count = len([m for m in chat.messages if not m.is_deleted])
        else:
            resp.message_count = 0

        return resp

    async def list_user_chats(self, user_id: int, skip: int = 0, limit: int = 100) -> List[ChatListItemDTO]:
        rows = await self.repo.get_list_items_with_stats(user_id=user_id, skip=skip, limit=limit)

        items: List[ChatListItemDTO] = []
        for chat, message_count, last_message_at in rows:
            dto = ChatListItemDTO.model_validate(chat)
            dto.message_count = int(message_count or 0)
            dto.last_message_at = last_message_at or chat.created_at
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
