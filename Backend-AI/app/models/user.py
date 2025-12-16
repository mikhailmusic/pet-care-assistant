from sqlalchemy import Column, String, Boolean, Text
from sqlalchemy.orm import relationship
from .base import BaseModel


class User(BaseModel):
    __tablename__ = "users"
    
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)

    google_credentials_json = Column(Text, nullable=True)
    
    # Relationships
    chats = relationship("Chat", back_populates="user")
    pets = relationship("Pet", back_populates="user")

    @property
    def has_google_calendar(self) -> bool:
        return self.google_credentials_json is not None
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"