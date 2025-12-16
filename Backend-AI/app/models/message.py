from sqlalchemy import Column, String, Integer, ForeignKey, Text, Enum, JSON
from sqlalchemy.orm import relationship
from .base import BaseModel
import enum


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    MIXED = "mixed"


class Message(BaseModel):
    __tablename__ = "messages"
    
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False, index=True)
    
    role = Column(Enum(MessageRole), nullable=False, index=True)
    content = Column(Text, nullable=False)
    message_type = Column(Enum(MessageType), default=MessageType.TEXT, nullable=False)
    files = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True) # Дополнительные данные (пути к файлам, данные о питомцах и т.д.)
    # Пример metadata:
    # {
    #   "files": [{"type": "image", "path": "minio://bucket/file.jpg", "name": "photo.jpg"}],
    #   "extracted_pet_info": {"name": "Барсик", "species": "собака"},
    #   "generated_content": [{"type": "image", "url": "..."}],
    #   "email_sent_to": "user@example.com",
    #   "web_search_used": true,
    #   "rag_sources": ["doc1.pdf", "doc2.pdf"]
    # }
    
    processing_time_ms = Column(Integer, nullable=True)  # Время обработки в миллисекундах
    
    chat = relationship("Chat", back_populates="messages")
    
    def __repr__(self):
        return f"<Message(id={self.id}, chat_id={self.chat_id}, role={self.role}, type={self.message_type})>"