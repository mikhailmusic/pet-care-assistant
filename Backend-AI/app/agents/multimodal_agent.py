from __future__ import annotations

from typing import Optional, Annotated
from datetime import datetime, timezone
from loguru import logger
import json
import io

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from app.integrations.gigachat_client import gigachat_client
from app.integrations import salutespeech_service
from app.integrations.minio_service import MinioService


class MultimodalTools:
    
    def __init__(self, minio_service: MinioService):
        self.minio_service = minio_service
    
    async def _get_file_from_minio(self, object_name: str) -> bytes:
        try:
            file_io = await self.minio_service.download_file(object_name)
            return file_io.getvalue()
        except Exception as e:
            logger.error(f"Failed to load file from MinIO: {object_name}, error: {e}")
            raise
    
    @tool
    async def analyze_image(
        self,
        state: Annotated[dict, InjectedState],
        question: str,
        file_index: int = 0,
    ) -> str:
        """Проанализировать изображение с помощью GigaChat Vision.
        
        Используй для:
        - Описания изображений питомцев
        - Анализа симптомов по фото
        - Идентификации породы
        - Оценки состояния здоровья по внешнему виду
        
        Args:
            state: Состояние графа (автоматически инжектится)
            question: Вопрос об изображении (например: "Опиши питомца", "Какие симптомы видны?")
            file_index: Индекс файла в uploaded_files (по умолчанию 0 - первый файл)
        
        Returns:
            JSON с анализом изображения
        """
        try:
            uploaded_files = state.get("uploaded_files", [])
            
            if not uploaded_files:
                return json.dumps({
                    "error": "Нет загруженных файлов для анализа"
                }, ensure_ascii=False)
            
            if file_index >= len(uploaded_files):
                return json.dumps({
                    "error": f"Неверный индекс файла: {file_index}. Доступно файлов: {len(uploaded_files)}"
                }, ensure_ascii=False)
            
            file_info = uploaded_files[file_index]
            object_name = file_info.get("object_name") or file_info.get("file_id")
            filename = file_info.get("filename", "unknown")
            file_type = file_info.get("file_type", "")
            mime_type = file_info.get("mime_type") or file_info.get("content_type")
            
            if not object_name:
                return json.dumps({
                    "error": f"Не удалось определить путь к файлу {filename}"
                }, ensure_ascii=False)
            
            # Проверяем тип файла
            if file_type.lower() != "image":
                return json.dumps({
                    "error": f"Файл '{filename}' не является изображением (тип: {file_type})"
                }, ensure_ascii=False)
            
            # Загружаем изображение из MinIO
            image_bytes = await self._get_file_from_minio(object_name)
            
            # Создаём BytesIO объект для vision_analysis
            image_io = io.BytesIO(image_bytes)
            
            # Анализируем через GigaChat Vision
            analysis_result = await gigachat_client.vision_analysis(
                file=image_io,
                filename=filename,
                prompt=question,
                mime_type=mime_type
            )
            
            result = {
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "file_ref": {
                    "filename": filename,
                    "object_name": object_name,
                    "file_type": file_type,
                    "file_size_bytes": len(image_bytes),
                    "mime_type": mime_type,
                },
                "question": question,
                "analysis": analysis_result,
                "model": "GigaChat Vision",
            }
            
            logger.info(f"Image analyzed: {filename}, question: {question[:50]}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return json.dumps({
                "error": str(e),
                "question": question
            }, ensure_ascii=False)
    
    @tool
    async def ocr_image(
        self,
        state: Annotated[dict, InjectedState],
        file_index: int = 0,
    ) -> str:
        """Извлечь текст из изображения (OCR) с помощью GigaChat Vision.
        
        Используй для:
        - Чтения текста с фотографий документов
        - Извлечения данных из медицинских справок
        - Распознавания текста с рецептов
        
        Args:
            state: Состояние графа (автоматически инжектится)
            file_index: Индекс файла в uploaded_files (по умолчанию 0 - первый файл)
        
        Returns:
            JSON с извлечённым текстом
        """
        try:
            uploaded_files = state.get("uploaded_files", [])
            
            if not uploaded_files:
                return json.dumps({
                    "error": "Нет загруженных файлов для OCR"
                }, ensure_ascii=False)
            
            if file_index >= len(uploaded_files):
                return json.dumps({
                    "error": f"Неверный индекс файла: {file_index}. Доступно файлов: {len(uploaded_files)}"
                }, ensure_ascii=False)
            
            file_info = uploaded_files[file_index]
            object_name = file_info.get("object_name") or file_info.get("file_id")
            filename = file_info.get("filename", "unknown")
            file_type = file_info.get("file_type", "")
            mime_type = file_info.get("mime_type") or file_info.get("content_type")
            
            if not object_name:
                return json.dumps({
                    "error": f"Не удалось определить путь к файлу {filename}"
                }, ensure_ascii=False)
            
            # Проверяем тип файла
            if file_type.lower() != "image":
                return json.dumps({
                    "error": f"Файл '{filename}' не является изображением (тип: {file_type})"
                }, ensure_ascii=False)
            
            # Загружаем изображение из MinIO
            image_bytes = await self._get_file_from_minio(object_name)
            
            # Создаём BytesIO объект
            image_io = io.BytesIO(image_bytes)
            
            # OCR промпт
            ocr_prompt = (
                "Извлеки весь текст с изображения. "
                "Сохрани форматирование и структуру. "
                "Если текста нет - напиши 'Текст не обнаружен'."
            )
            
            # OCR через GigaChat Vision
            extracted_text = await gigachat_client.vision_analysis(
                file=image_io,
                filename=filename,
                prompt=ocr_prompt,
                mime_type=mime_type
            )
            
            result = {
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "file_ref": {
                    "filename": filename,
                    "object_name": object_name,
                    "file_type": file_type,
                    "file_size_bytes": len(image_bytes),
                    "mime_type": mime_type,
                },
                "operation": "ocr",
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "model": "GigaChat Vision OCR",
            }
            
            logger.info(f"OCR completed: {filename}, extracted {len(extracted_text)} characters")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to perform OCR: {e}")
            return json.dumps({
                "error": str(e)
            }, ensure_ascii=False)
    
    @tool
    async def transcribe_audio(
        self,
        state: Annotated[dict, InjectedState],
        file_index: int = 0,
    ) -> str:
        """Транскрибировать аудио в текст с помощью SaluteSpeech STT.
        
        Используй для:
        - Преобразования голосовых сообщений в текст
        - Транскрипции аудио-записей
        
        Args:
            state: Состояние графа (автоматически инжектится)
            file_index: Индекс файла в uploaded_files (по умолчанию 0 - первый файл)
        
        Returns:
            JSON с транскрибированным текстом
        """
        try:
            uploaded_files = state.get("uploaded_files", [])
            
            if not uploaded_files:
                return json.dumps({
                    "error": "Нет загруженных аудио для транскрибации"
                }, ensure_ascii=False)
            
            if file_index >= len(uploaded_files):
                return json.dumps({
                    "error": f"Неверный индекс файла: {file_index}. Доступно файлов: {len(uploaded_files)}"
                }, ensure_ascii=False)
            
            file_info = uploaded_files[file_index]
            object_name = file_info.get("object_name") or file_info.get("file_id")
            filename = file_info.get("filename", "unknown")
            file_type = file_info.get("file_type", "")
            mime_type = file_info.get("mime_type") or file_info.get("content_type")
            
            if not object_name:
                return json.dumps({
                    "error": f"Не удалось определить путь к файлу {filename}"
                }, ensure_ascii=False)
            
            # Проверяем тип файла
            if file_type.lower() != "audio":
                return json.dumps({
                    "error": f"Файл '{filename}' не является аудио (тип: {file_type})"
                }, ensure_ascii=False)
            
            # Загружаем аудио из MinIO
            audio_bytes = await self._get_file_from_minio(object_name)
            
            # Транскрибируем через SaluteSpeech
            transcription = await salutespeech_service.speech_to_text(audio_bytes)
            
            result = {
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "file_ref": {
                    "filename": filename,
                    "object_name": object_name,
                    "file_type": file_type,
                    "file_size_bytes": len(audio_bytes),
                    "mime_type": mime_type,
                },
                "operation": "speech_to_text",
                "transcription": transcription,
                "transcription_length": len(transcription),
                "model": "SaluteSpeech STT",
            }
            
            logger.info(f"Audio transcribed: {filename}, {len(transcription)} characters")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return json.dumps({
                "error": str(e)
            }, ensure_ascii=False)
    
    @tool
    async def analyze_video(
        self,
        state: Annotated[dict, InjectedState],
        question: str,
        file_index: int = 0,
        frame_interval: int = 30,
        max_frames: int = 5,
    ) -> str:
        """Проанализировать видео (извлекает ключевые кадры и анализирует их).
        
        Используй для:
        - Анализа поведения питомца на видео
        - Оценки активности
        - Поиска аномалий в движениях
        
        Args:
            state: Состояние графа (автоматически инжектится)
            question: Вопрос о видео
            file_index: Индекс файла в uploaded_files (по умолчанию 0 - первый файл)
            frame_interval: Интервал извлечения кадров в секундах (по умолчанию 30)
            max_frames: Максимальное количество кадров для анализа (по умолчанию 5)
        
        Returns:
            JSON с анализом видео
        """
        try:
            uploaded_files = state.get("uploaded_files", [])
            
            if not uploaded_files:
                return json.dumps({
                    "error": "Нет загруженных видео для анализа"
                }, ensure_ascii=False)
            
            if file_index >= len(uploaded_files):
                return json.dumps({
                    "error": f"Неверный индекс файла: {file_index}. Доступно файлов: {len(uploaded_files)}"
                }, ensure_ascii=False)
            
            file_info = uploaded_files[file_index]
            object_name = file_info.get("object_name") or file_info.get("file_id")
            filename = file_info.get("filename", "unknown")
            file_type = file_info.get("file_type", "")
            mime_type = file_info.get("mime_type") or file_info.get("content_type")
            
            if not object_name:
                return json.dumps({
                    "error": f"Не удалось определить путь к файлу {filename}"
                }, ensure_ascii=False)
            
            # Проверяем тип файла
            if file_type.lower() != "video":
                return json.dumps({
                    "error": f"Файл '{filename}' не является видео (тип: {file_type})"
                }, ensure_ascii=False)
            
            # Загружаем видео из MinIO
            video_bytes = await self._get_file_from_minio(object_name)
            
            # Извлекаем ключевые кадры
            import cv2
            import numpy as np
            import tempfile
            import os
            
            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name
            
            try:
                # Открываем видео
                cap = cv2.VideoCapture(tmp_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                # Извлекаем кадры с заданным интервалом
                extracted_frames = []
                frame_timestamps = []
                
                interval_frames = frame_interval * fps
                current_frame = 0
                
                while cap.isOpened() and len(extracted_frames) < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if current_frame % interval_frames == 0:
                        # Конвертируем в JPEG
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        extracted_frames.append(frame_bytes)
                        frame_timestamps.append(current_frame / fps)
                    
                    current_frame += 1
                
                cap.release()
                
                if not extracted_frames:
                    return json.dumps({
                        "error": f"Не удалось извлечь кадры из видео {filename}"
                    }, ensure_ascii=False)
                
                # Анализируем кадры через GigaChat Vision (multiple)
                # Подготавливаем файлы для vision_analysis_multiple
                files_for_analysis = []
                for i, (frame_bytes, timestamp) in enumerate(zip(extracted_frames, frame_timestamps)):
                    frame_io = io.BytesIO(frame_bytes)
                    frame_filename = f"frame_{i}_at_{timestamp:.1f}s.jpg"
                    files_for_analysis.append((frame_io, frame_filename, "image/jpeg"))
                
                # Анализируем все кадры сразу
                combined_prompt = (
                    f"{question}\n\n"
                    f"Проанализируй {len(extracted_frames)} кадров из видео. "
                    f"Каждый кадр взят с интервалом {frame_interval} секунд. "
                    f"Опиши общее поведение и изменения на протяжении видео."
                )
                
                analysis_result = await gigachat_client.vision_analysis_multiple(
                    files=files_for_analysis,
                    prompt=combined_prompt
                )
                
                result = {
                    "analyzed_at": datetime.now(timezone.utc).isoformat(),
                    "file_ref": {
                        "filename": filename,
                        "object_name": object_name,
                        "file_type": file_type,
                        "file_size_bytes": len(video_bytes),
                        "mime_type": mime_type,
                    },
                    "operation": "video_analysis",
                    "question": question,
                    "video_metadata": {
                        "duration_seconds": duration,
                        "fps": fps,
                        "total_frames": frame_count,
                    },
                    "analysis_metadata": {
                        "frames_analyzed": len(extracted_frames),
                        "frame_interval_seconds": frame_interval,
                        "frame_timestamps": frame_timestamps,
                    },
                    "analysis": analysis_result,
                    "model": "GigaChat Vision (Multiple)",
                }
                
                logger.info(f"Video analyzed: {filename}, {len(extracted_frames)} frames")
                return json.dumps(result, ensure_ascii=False, indent=2)
                
            finally:
                # Удаляем временный файл
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Failed to analyze video: {e}")
            return json.dumps({
                "error": str(e),
                "question": question
            }, ensure_ascii=False)


def create_multimodal_agent(
    minio_service: MinioService,
    llm,
    name: str = "multimodal",
):
    """Создать агента для анализа мультимедиа (изображения, аудио, видео)
    
    Args:
        minio_service: Сервис для загрузки файлов из MinIO
        llm: Языковая модель
        name: Имя агента (для supervisor handoff)
    
    Returns:
        Compiled ReAct agent
    """
    tools_instance = MultimodalTools(minio_service)
    
    tools = [
        tools_instance.analyze_image,
        tools_instance.ocr_image,
        tools_instance.transcribe_audio,
        tools_instance.analyze_video,
    ]
    
    prompt = (
        "Ты - специалист по анализу мультимедиа.\n\n"
        "Твои возможности:\n"
        "- Анализ изображений (GigaChat Vision)\n"
        "- OCR - извлечение текста из изображений\n"
        "- Транскрибация аудио (SaluteSpeech STT)\n"
        "- Анализ видео (извлечение ключевых кадров)\n\n"
        "Рабочий процесс:\n"
        "1. Проверь uploaded_files в state - там список загруженных файлов\n"
        "2. Используй file_index для выбора нужного файла (по умолчанию 0)\n"
        "3. Для изображений: analyze_image или ocr_image\n"
        "4. Для аудио: transcribe_audio\n"
        "5. Для видео: analyze_video\n\n"
        "Все файлы автоматически загружаются из MinIO.\n"
        "Верни JSON с результатами анализа."
    )
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        name=name,
        prompt=prompt,
    )
    
    logger.info(f"Created MultimodalAgent '{name}' with {len(tools)} tools")
    return agent