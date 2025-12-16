from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, ConfigDict
from .common import TimestampDTO
from app.models.message import MessageRole, MessageType


class MessageCreateDTO(BaseModel):    
    content: str = Field(..., min_length=1)
    files: Optional[List[str]] = None  # Список ID загруженных файлов (из MinIO)
    

class MessageUpdateDTO(BaseModel):    
    content: str = Field(..., min_length=1)
    files: Optional[List[str]] = None
    

class MessageResponseDTO(TimestampDTO):    
    chat_id: int
    role: MessageRole
    content: str
    message_type: MessageType
    files: Optional[List[Dict[str, Any]]] = None    
    metadata_json: Optional[Dict[str, Any]] = None
    

class StreamMessageChunkDTO(BaseModel):    
    chunk: str
    is_final: bool = False
    metadata_json: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)