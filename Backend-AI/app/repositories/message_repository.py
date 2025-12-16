from datetime import datetime
from typing import List
from sqlalchemy import select, exists
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Message, Chat
from app.repositories.base_repository import BaseRepository


class MessageRepository(BaseRepository[Message]):
    
    def __init__(self, db: AsyncSession):
        super().__init__(Message, db)
    
    async def get_chat_messages(self, chat_id: int, skip: int | None = None, limit: int | None = None, include_deleted: bool = False, order_desc: bool = False) -> List[Message]:
        query = select(Message).where(Message.chat_id == chat_id)
        query = self._apply_deleted_filter(query, include_deleted)

        # Сортировка в БД (ВАЖНО: до limit!)
        if order_desc:
            query = query.order_by(Message.created_at.desc())
        else:
            query = query.order_by(Message.created_at.asc())

        if skip is not None:
            query = query.offset(skip)
        if limit is not None:
            query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())



    async def get_chat_messages_before_date(self,chat_id: int,before: datetime,limit: int | None = None,include_deleted: bool = False) -> List[Message]:
        query = select(Message).where(
            Message.chat_id == chat_id,
            Message.created_at < before
        )

        query = self._apply_deleted_filter(query, include_deleted)

        if limit is not None:
            query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_chat_messages_after_date(self,chat_id: int,after: datetime,limit: int | None = None,include_deleted: bool = False) -> List[Message]:
        query = select(Message).where(
            Message.chat_id == chat_id,
            Message.created_at > after
        )

        query = self._apply_deleted_filter(query, include_deleted)

        if limit is not None:
            query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def delete_messages_after(self, chat_id: int, after_message_id: int) -> int:
        """Удаляет (soft delete) все сообщения в чате после указанного message_id."""
        # Получаем timestamp сообщения
        msg = await self.get_by_id(after_message_id)
        if not msg or msg.chat_id != chat_id:
            return 0

        # Находим все сообщения после этого времени
        query = select(Message).where(
            Message.chat_id == chat_id,
            Message.created_at > msg.created_at,
            Message.is_deleted.is_(False)
        )
        result = await self.db.execute(query)
        messages_to_delete = list(result.scalars().all())

        # Soft delete для каждого
        for m in messages_to_delete:
            m.soft_delete()

        await self.db.flush()
        return len(messages_to_delete)

    async def file_belongs_to_user_chats(self, user_id: int, file_id: str) -> bool:
        stmt = select(
            exists().where(
                Message.chat_id == Chat.id,
            ).where(
                Chat.user_id == user_id,
            ).where(
                Message.is_deleted.is_(False),
            ).where(
                Message.files.contains([{"file_id": file_id}])
            )
        ).select_from(Message).join(Chat, Chat.id == Message.chat_id)

        res = await self.db.execute(stmt)
        return bool(res.scalar())