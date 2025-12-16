from datetime import datetime, timezone
from sqlalchemy import Integer, Boolean, DateTime, Column
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class BaseModel(Base):    
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    def soft_delete(self):
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)

    def restore(self):
        self.is_deleted = False
        self.deleted_at = None

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id})>"