from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[User]):
    
    def __init__(self, db: AsyncSession):
        super().__init__(User, db)
    
    async def get_by_email(self, email: str, include_deleted: bool = False) -> Optional[User]:
        query = select(User).where(User.email == email)
        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return result.scalars().first()
    
    async def email_exists(self, email: str, exclude_user_id: Optional[int] = None, include_deleted: bool = False) -> bool:
        query = select(User.id).where(User.email == email)
        query = self._apply_deleted_filter(query, include_deleted)

        if exclude_user_id is not None:
            query = query.where(User.id != exclude_user_id)
        
        result = await self.db.execute(query)
        return result.scalar() is not None
