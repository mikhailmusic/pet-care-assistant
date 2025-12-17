from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .common import TimestampDTO
from app.models.message import MessageRole, MessageType


class _MessageBaseDTO(BaseModel):
    content: Optional[str] = Field(None, min_length=0)
    files: Optional[List[str]] = None  # öõñ‘??ó ID úø?‘?‘?ç??‘<‘: ‘"øü>?? (ñú MinIO)

    @model_validator(mode="after")
    def _require_text_or_files(self):
        has_content = bool(self.content and self.content.strip())
        has_files = bool(self.files)
        if not has_content and not has_files:
            raise ValueError("Either content or files must be provided")

        # Normalize to empty string so DB column stays non-null
        if self.content is None:
            self.content = ""
        return self


class MessageCreateDTO(_MessageBaseDTO):
    ...


class MessageUpdateDTO(_MessageBaseDTO):
    ...


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
