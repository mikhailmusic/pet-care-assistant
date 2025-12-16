from typing import List
from datetime import date
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.health_record import HealthRecord, RecordType
from app.repositories.base_repository import BaseRepository


class HealthRecordRepository(BaseRepository[HealthRecord]):
    
    def __init__(self, db: AsyncSession):
        super().__init__(HealthRecord, db)
    
    async def get_by_pet_id(self,pet_id: int,record_type: RecordType | None = None,include_deleted: bool = False) -> List[HealthRecord]:

        query = select(HealthRecord).where(HealthRecord.pet_id == pet_id)

        if record_type is not None:
            query = query.where(HealthRecord.record_type == record_type)

        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return list(result.scalars().all())
        
    async def get_by_date_range(self, pet_id: int, start_date: date, end_date: date, include_deleted: bool = False) -> List[HealthRecord]:

        query = select(HealthRecord).where(
            HealthRecord.pet_id == pet_id,
            HealthRecord.record_date >= start_date,
            HealthRecord.record_date <= end_date
        )

        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return list(result.scalars().all())
    
    async def get_by_type(self,pet_id: int,record_type: RecordType,include_deleted: bool = False,) -> List[HealthRecord]:

        query = select(HealthRecord).where(
            HealthRecord.pet_id == pet_id,
            HealthRecord.record_type == record_type,
        )
        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_unresolved(self,pet_id: int,include_deleted: bool = False,) -> List[HealthRecord]:

        query = select(HealthRecord).where(
            HealthRecord.pet_id == pet_id,
            HealthRecord.is_resolved.is_(False),
        )
        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return list(result.scalars().all())
