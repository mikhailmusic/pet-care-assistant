from typing import List, Optional
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.pet import Pet
from app.repositories.base_repository import BaseRepository


class PetRepository(BaseRepository[Pet]):    
    def __init__(self, db: AsyncSession):
        super().__init__(Pet, db)
    
    async def get_by_user_id(self, user_id: int, include_deleted: bool = False) -> List[Pet]:
        query = select(Pet).where(Pet.user_id == user_id)
        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_with_health_records(self, pet_id: int, include_deleted: bool = False) -> Optional[Pet]:
        query = select(Pet).where(Pet.id == pet_id)
        query = self._apply_deleted_filter(query, include_deleted)
        query = query.options(selectinload(Pet.health_records))
        
        result = await self.db.execute(query)
        return result.scalars().first()
    
    async def find_by_name(self, user_id: int, name: str, include_deleted: bool = False) -> Optional[Pet]:
        query = select(Pet).where(
            Pet.user_id == user_id,
            Pet.name.ilike(f"%{name}%")
        )
        query = self._apply_deleted_filter(query, include_deleted)
        
        result = await self.db.execute(query)
        return result.scalars().first()
    
    async def search_pets(self, user_id: int, search_query: str, include_deleted: bool = False) -> List[Pet]:
        query = select(Pet).where(
            Pet.user_id == user_id,
            or_(
                Pet.name.ilike(f"%{search_query}%"),
                Pet.species.ilike(f"%{search_query}%"),
                Pet.breed.ilike(f"%{search_query}%")
            )
        )
        query = self._apply_deleted_filter(query, include_deleted)
        
        result = await self.db.execute(query)
        return list(result.scalars().all())
