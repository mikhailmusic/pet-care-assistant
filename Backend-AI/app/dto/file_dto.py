from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class FileUploadResponseDTO(BaseModel):    
    file_id: str  # Уникальный ID файла
    file_name: str
    file_type: str  # image, video, audio, document
    file_size: int  # Размер в байтах
    mime_type: str
    url: str  # URL для доступа к файлу
    thumbnail_url: Optional[str] = None  # Для изображений и видео
    uploaded_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class MultipleFileUploadResponseDTO(BaseModel):    
    files: List[FileUploadResponseDTO]
    total_size: int  # Общий размер всех файлов
    
    model_config = ConfigDict(from_attributes=True)


class FileMetadataDTO(BaseModel):
    file_id: str  # object_name в MinIO
    file_name: str
    file_type: str  # "image", "video", "audio", "document"
    mime_type: str
    file_size: int
    url: str
    
    model_config = ConfigDict(from_attributes=True)
