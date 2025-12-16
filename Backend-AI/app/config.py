from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from functools import lru_cache


class Settings(BaseSettings): 
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    DATABASE_URL: str
    
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    
    GIGACHAT_API_KEY: str
    GIGACHAT_SCOPE: str = "GIGACHAT_API_PERS"
    GIGACHAT_MODEL: str = "GigaChat-Lite"
    GIGACHAT_VERIFY_SSL_CERTS: bool = False
    GIGACHAT_TEMPERATURE: float = 0.7
    
    SALUTESPEECH_API_KEY: str
    
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET_NAME: str = "petcare-files"
    MINIO_SECURE: bool = False
    
    SMTP_HOST: str
    SMTP_PORT: int = 587
    SMTP_USERNAME: str
    SMTP_PASSWORD: str
    SMTP_USE_TLS: bool = True
    
    CHROMA_PERSIST_DIRECTORY: str = "./vector_db"
    CHROMA_COLLECTION_NAME: str = "petcare_documents"
    
    APP_NAME: str = "PetCare AI Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    GOOGLE_CALENDAR_CREDENTIALS_FILE: str = "credentials/google_calendar_credentials.json"
    GOOGLE_CALENDAR_TOKEN_FILE: str = "credentials/google_calendar_token.json"
        
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    

    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_IMAGE_EXTENSIONS: str = "jpg,jpeg,png,webp,gif"
    ALLOWED_VIDEO_EXTENSIONS: str = "mp4,avi,mov,mkv"
    ALLOWED_AUDIO_EXTENSIONS: str = "mp3,wav,ogg,m4a"
    ALLOWED_DOCUMENT_EXTENSIONS: str = "pdf,docx,txt,csv,xlsx,xls"


    # RAG Settings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_DEVICE: str = "cpu"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    
    DDG_MAX_RESULTS: int = 10
    DDG_REGION: str = "ru-ru"
    DDG_SAFESEARCH: str = "moderate"
    DDG_TIMELIMIT: str = "y"

    @property
    def cors_origins_list(self) -> List[str]:
        """Преобразование строки CORS origins в список"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def allowed_image_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_IMAGE_EXTENSIONS.split(",")]
    
    @property
    def allowed_video_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_VIDEO_EXTENSIONS.split(",")]
    
    @property
    def allowed_audio_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_AUDIO_EXTENSIONS.split(",")]
    
    @property
    def allowed_document_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_DOCUMENT_EXTENSIONS.split(",")]
    
    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Singleton экземпляра настроек (кешируется)"""
    return Settings()

# Глобальный экземпляр настроек
settings = get_settings()
