from minio import Minio
from minio.error import S3Error
from typing import BinaryIO
from io import BytesIO
from datetime import timedelta
import uuid
import asyncio
from loguru import logger

from app.config import settings
from app.utils.exceptions import MinIOException, FileNotFoundException


class MinioService:
    def __init__(self):
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self.bucket_name = settings.MINIO_BUCKET_NAME

    async def ensure_bucket_exists(self) -> bool:
        def _check_and_create():
            try:
                if not self.client.bucket_exists(self.bucket_name):
                    self.client.make_bucket(self.bucket_name)
                    return True  # Bucket создан
                return False     # Bucket уже существует
            except S3Error as e:
                raise MinIOException(f"Failed to ensure MinIO bucket exists: {e}") from e

        return await asyncio.to_thread(_check_and_create)

    async def upload_file(self,file: BinaryIO,filename: str, content_type: str,folder: str = "uploads",) -> str:
        # Генерация уникального имени файла
        file_extension = filename.split(".")[-1] if "." in filename else ""
        unique_filename = (
            f"{filename}-{uuid.uuid4()}.{file_extension}" if file_extension else str(uuid.uuid4())
        )
        object_name = f"{folder}/{unique_filename}"

        # Получение размера файла
        file.seek(0, 2)  # Перемещение в конец файла
        file_size = file.tell()
        file.seek(0)  # Возврат в начало

        def _put_object():
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=file,
                length=file_size,
                content_type=content_type,
                metadata={"original_filename": filename}
            )

        try:
            await asyncio.to_thread(_put_object)
            return object_name
        except S3Error as e:
            raise MinIOException(f"Failed to upload file to MinIO: {e}")

    async def download_file(self, object_name: str) -> BytesIO:
        def _download() -> bytes:
            response = self.client.get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
            )
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()
            return data

        try:
            data = await asyncio.to_thread(_download)
            return BytesIO(data)
        except S3Error as e:
            if getattr(e, "code", "") == "NoSuchKey":
                raise FileNotFoundException(object_name) from e
            raise MinIOException(f"Failed to download file from MinIO: {e}") from e

    async def delete_file(self, object_name: str) -> bool:
        def _delete():
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
            )

        try:
            await asyncio.to_thread(_delete)
            return True
        except S3Error as e:
            logger.error(f"Failed to delete file from MinIO: {e}")
            return False

    async def file_exists(self, object_name: str) -> bool:
        def _stat():
            self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
            )

        try:
            await asyncio.to_thread(_stat)
            return True
        except S3Error:
            return False

    async def get_presigned_url(self, object_name: str, expires: timedelta = timedelta(hours=1)) -> str:
        def _presigned() -> str:
            return self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=expires,
            )

        try:
            return await asyncio.to_thread(_presigned)
        except S3Error as e:
            if getattr(e, "code", "") == "NoSuchKey":
                raise FileNotFoundException(object_name) from e
            raise MinIOException(f"Failed to generate presigned URL: {e}") from e

    async def get_file_url(self, object_name: str) -> str:
        protocol = "https" if settings.MINIO_SECURE else "http"
        return f"{protocol}://{settings.MINIO_ENDPOINT}/{self.bucket_name}/{object_name}"

    async def list_files(self, prefix: str = "") -> list[str]:
        def _list() -> list[str]:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True,
            )
            return [obj.object_name for obj in objects]

        try:
            return await asyncio.to_thread(_list)
        except S3Error as e:
            raise MinIOException(f"Failed to list files in MinIO: {e}") from e

    async def get_file_metadata(self, object_name: str) -> dict:
        def _stat():
            return self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
            )
        
        try:
            stat = await asyncio.to_thread(_stat)
            return {
                "object_name": object_name,
                "size": stat.size,
                "content_type": stat.content_type,
                "original_filename": stat.metadata.get("x-amz-meta-original_filename"),
                "last_modified": stat.last_modified,
            }
        except S3Error as e:
            raise MinIOException(f"Failed to get metadata: {e}")


minio_service = MinioService()
