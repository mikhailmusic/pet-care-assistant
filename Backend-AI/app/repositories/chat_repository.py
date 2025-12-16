from typing import List
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Chat, Message
from app.repositories.base_repository import BaseRepository


class ChatRepository(BaseRepository[Chat]):
    
    def __init__(self, db: AsyncSession):
        super().__init__(Chat, db)
    
    async def get_by_user_id(self,user_id: int, include_deleted: bool = False) -> List[Chat]:
        query = select(Chat).where(Chat.user_id == user_id)

        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_with_messages(self,chat_id: int,include_deleted: bool = False) -> Chat | None:
        query = select(Chat).where(Chat.id == chat_id)
        query = self._apply_deleted_filter(query, include_deleted)
        query = query.options(selectinload(Chat.messages))

        result = await self.db.execute(query)
        return result.scalars().first()
    
    async def get_list_items_with_stats(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        include_deleted: bool = False,
    ):
        """
        Возвращает список кортежей:
          (Chat, message_count, last_message_at)
        message_count учитывает только НЕудалённые сообщения.
        last_message_at = max(created_at) по сообщениям (если нет — None).
        """
        msg_count = func.count(Message.id)
        last_msg_at = func.max(Message.created_at)

        query = (
            select(
                Chat,
                msg_count.label("message_count"),
                last_msg_at.label("last_message_at"),
            )
            .outerjoin(
                Message,
                (Message.chat_id == Chat.id) & (Message.is_deleted.is_(False)),
            )
            .where(Chat.user_id == user_id)
            .group_by(Chat.id)
            .order_by(func.coalesce(last_msg_at, Chat.updated_at).desc())
            .offset(skip)
            .limit(limit)
        )

        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return list(result.all())