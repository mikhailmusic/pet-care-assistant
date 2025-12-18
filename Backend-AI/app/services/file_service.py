# app/services/file_service.py

from __future__ import annotations

from typing import List, Optional, Tuple
from datetime import datetime
from loguru import logger

from fastapi import UploadFile

from app.dto import (
    FileUploadResponseDTO,
    MultipleFileUploadResponseDTO,
    FileMetadataDTO,
)
from app.integrations.minio_service import MinioService
from app.utils.exceptions import MinIOException, FileNotFoundException, AuthorizationException


def _infer_file_type(mime_type: str, filename: Optional[str] = None) -> str:
    """
    Определить тип файла по MIME-типу и расширению.

    Args:
        mime_type: MIME-тип файла
        filename: Имя файла (опционально, для fallback по расширению)

    Returns:
        Тип файла: "image", "video", "audio", или "document"
    """
    mt = (mime_type or "").lower()

    # Попытка 1: Определение по MIME-типу
    if mt.startswith("image/"):
        return "image"
    if mt.startswith("video/"):
        return "video"
    if mt.startswith("audio/"):
        return "audio"

    # Попытка 2: Если MIME-тип неизвестен, пробуем определить по расширению
    if filename and mt in ("application/octet-stream", "binary/octet-stream", ""):
        ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ""

        # Изображения
        if ext in ('jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg', 'ico', 'tiff', 'tif'):
            return "image"

        # Видео
        if ext in ('mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'webm', 'm4v', 'mpg', 'mpeg'):
            return "video"

        # Аудио
        if ext in ('mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'wma', 'opus'):
            return "audio"

    return "document"


class FileService:
    """Сервис загрузки/получения файлов для фронтенда (MinIO)."""

    def __init__(self, minio_service: MinioService):
        self.minio = minio_service

    async def upload_file(
        self,
        user_id: int,
        file: UploadFile,
        folder: str = "uploads",
    ) -> FileUploadResponseDTO:
        """
        Загружает файл в MinIO и возвращает FileUploadResponseDTO.
        file_id == object_name в MinIO.
        """
        try:
            content = await file.read()
            file_size = len(content)
            mime_type = file.content_type or "application/octet-stream"
            file_type = _infer_file_type(mime_type, file.filename)

            # MinioService.upload_file ожидает BinaryIO => используем BytesIO
            from io import BytesIO
            bio = BytesIO(content)

            object_name = await self.minio.upload_file(
                file=bio,
                filename=file.filename or "file",
                content_type=mime_type,
                folder=folder,
            )

            url = await self.minio.get_file_url(object_name)

            dto = FileUploadResponseDTO(
                file_id=object_name,
                file_name=file.filename or "file",
                file_type=file_type,
                file_size=file_size,
                mime_type=mime_type,
                url=url,
                thumbnail_url=None,
                uploaded_at=datetime.utcnow(),
            )

            logger.info(f"User {user_id} uploaded file -> {object_name} ({mime_type}, {file_size}b)")
            return dto

        except Exception as e:
            logger.exception(f"Upload file failed for user {user_id}")
            raise MinIOException(f"Failed to upload file: {e}") from e

    async def upload_files(
        self,
        user_id: int,
        files: List[UploadFile],
        folder: str = "uploads",
    ) -> MultipleFileUploadResponseDTO:
        """Массовая загрузка."""
        uploaded: List[FileUploadResponseDTO] = []
        total = 0

        for f in files:
            dto = await self.upload_file(user_id=user_id, file=f, folder=folder)
            uploaded.append(dto)
            total += dto.file_size

        return MultipleFileUploadResponseDTO(files=uploaded, total_size=total)

    async def get_file_metadata(
        self,
        user_id: int,
        file_id: str,
    ) -> FileMetadataDTO:
        """Метаданные файла по object_name."""
        stat = await self.minio.get_file_metadata(file_id)
        url = await self.minio.get_file_url(file_id)

        mime_type = stat.get("content_type") or "application/octet-stream"
        filename = stat.get("original_filename") or file_id.split("/")[-1]
        file_type = _infer_file_type(mime_type, filename)

        return FileMetadataDTO(
            file_id=file_id,
            file_name=filename,
            file_type=file_type,
            mime_type=mime_type,
            file_size=int(stat.get("size") or 0),
            url=url,
        )

    async def delete_file(self, user_id: int, file_id: str) -> bool:

        exists = await self.minio.file_exists(file_id)
        if not exists:
            raise FileNotFoundException(file_id)

        # 2) Проверка владения через БД
        allowed = await self.message_repo.file_belongs_to_user_chats(user_id=user_id, file_id=file_id)
        if not allowed:
            raise AuthorizationException("Нельзя удалить файл: он не найден в ваших чатах/сообщениях")

        # 3) Удаление
        return await self.minio.delete_file(file_id)