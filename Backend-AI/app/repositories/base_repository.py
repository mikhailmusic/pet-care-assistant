from typing import TypeVar, Generic, Type, Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models import BaseModel

ModelType = TypeVar("ModelType", bound=BaseModel)


class BaseRepository(Generic[ModelType]):
    """Базовый репозиторий для работы с моделями.
    
    Предоставляет базовые CRUD операции.
    Логическое удаление (is_deleted) обрабатывается в сервисах или моделях.
    """

    def __init__(self, model: Type[ModelType], db: AsyncSession):
        self.model = model
        self.db = db

    def _apply_deleted_filter(self, query, include_deleted: bool):
        if not include_deleted:
            query = query.where(self.model.is_deleted.is_(False))
        return query

    async def create(self, db_obj: ModelType) -> ModelType:
        self.db.add(db_obj)
        try:
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            raise

        await self.db.refresh(db_obj)
        return db_obj

    async def get_by_id(self, id: int, include_deleted: bool = False) -> Optional[ModelType]:
        query = select(self.model).where(self.model.id == id)
        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return result.scalars().first()

    async def get_all(self, skip: int = 0, limit: int = 100, include_deleted: bool = False) -> List[ModelType]:
        query = select(self.model)
        query = self._apply_deleted_filter(query, include_deleted)
        query = query.offset(skip).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def update(self, db_obj: ModelType) -> ModelType:
        try:
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            raise

        await self.db.refresh(db_obj)
        return db_obj

    async def exists(self, id: int, include_deleted: bool = False) -> bool:
        query = select(self.model.id).where(self.model.id == id)
        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return result.scalar() is not None

    async def count(self, include_deleted: bool = False) -> int:
        query = select(func.count()).select_from(self.model)
        query = self._apply_deleted_filter(query, include_deleted)

        result = await self.db.execute(query)
        return result.scalar() or 0
