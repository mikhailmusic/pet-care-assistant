from sqlalchemy import Column, String, Integer, Boolean, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from .base import BaseModel


class Chat(BaseModel):
    __tablename__ = "chats"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    title = Column(String(255), nullable=False, default="Новый чат")
    description = Column(Text, nullable=True)
    
    # Settings (управляются пользователем из UI)
    web_search_enabled = Column(Boolean, default=True, nullable=False)  # Включить/выключить веб-поиск
    message_limit = Column(Integer, default=20, nullable=True)  # Ограничение кол-ва учитываемых сообщений (None = все)
    temperature = Column(Float, default=0.7, nullable=False)  # Температура генерации (0.0-1.0)
    gigachat_model = Column(String(50), default="GigaChat-Lite", nullable=False)  # GigaChat-Lite/Pro/Max
    image_generation_enabled = Column(Boolean, default=True, nullable=False)  # Генерация изображений
    voice_response_enabled = Column(Boolean, default=False, nullable=False)  # TTS для ответов
    
    max_tokens = Column(Integer, default=2000, nullable=False)  # Максимум токенов в ответе
    

    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat")
    
    def __repr__(self):
        return f"<Chat(id={self.id}, title={self.title}, user_id={self.user_id})>"