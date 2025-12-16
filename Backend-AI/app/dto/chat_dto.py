from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from .common import TimestampDTO


class ChatCreateDTO(BaseModel):    
    title: str = Field(default="Новый чат", max_length=255)
    description: Optional[str] = None


class ChatUpdateDTO(BaseModel):    
    title: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    
    # Settings
    web_search_enabled: Optional[bool] = None
    message_limit: Optional[int] = Field(None, ge=1, le=100)
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    gigachat_model: Optional[str] = Field(None, max_length=50)
    image_generation_enabled: Optional[bool] = None
    voice_response_enabled: Optional[bool] = None
    max_tokens: Optional[int] = Field(None, ge=100, le=10000)

    

class ChatResponseDTO(TimestampDTO):    
    user_id: int
    title: str
    description: Optional[str] = None
    
    # Settings
    web_search_enabled: bool
    message_limit: Optional[int]
    temperature: float
    gigachat_model: str
    image_generation_enabled: bool
    voice_response_enabled: bool
    max_tokens: int
    
    # Статистика
    message_count: Optional[int] = 0
    

class ChatListItemDTO(TimestampDTO):
    
    user_id: int
    title: str
    description: Optional[str] = None
    message_count: Optional[int] = 0
    last_message_at: Optional[datetime] = None


class ChatSettingsDTO(BaseModel):
    web_search_enabled: bool
    message_limit: Optional[int] = Field(None, ge=1, le=100)
    temperature: float = Field(..., ge=0.0, le=1.0)
    gigachat_model: str
    image_generation_enabled: bool
    voice_response_enabled: bool
    max_tokens: int = Field(..., ge=100, le=10000)

    model_config = ConfigDict(from_attributes=True)