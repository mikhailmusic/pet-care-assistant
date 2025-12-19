# app/agents/multimodal_agent.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal, BinaryIO
from datetime import datetime, timezone
from loguru import logger
from contextvars import ContextVar
import json
import io
import base64

from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.integrations.gigachat_client import gigachat_client
from app.integrations import salutespeech_service
from app.integrations.minio_service import MinioService
from app.integrations import minio_service as MinioServiceDep
from app.config import settings


@dataclass
class MultimodalContext:
    """Контекст для Multimodal Agent"""
    user_id: int
    uploaded_files: List[Dict[str, Any]]


_multimodal_context: ContextVar[Optional[MultimodalContext]] = ContextVar(
    '_multimodal_context',
    default=None
)

_minio_service: ContextVar[Optional[MinioService]] = ContextVar('_minio_service', default=None)



def _get_context() -> MultimodalContext:
    """Get the current context from ContextVar"""
    ctx = _multimodal_context.get()
    if ctx is None:
        raise RuntimeError("Multimodal context not set.")
    return ctx


def _get_minio_service() -> MinioService:
    service = _minio_service.get()
    if service is None:
        raise RuntimeError("Minio service not set.")
    return service


async def _get_file_from_ref(file_ref: Optional[str]) -> tuple[BinaryIO, str, str]:
    """
    Returns: (file_object(BytesIO), filename, mime_type)
    """
    ctx = _get_context()
    minio_service = _get_minio_service()

    logger.debug(f"_get_file_from_ref called with file_ref={file_ref}, uploaded_files count={len(ctx.uploaded_files)}")

    if not file_ref:
        # Случай 1: file_ref не указан - берем первый файл
        if not ctx.uploaded_files:
            raise ValueError("Нет загруженных файлов. Укажи file_ref или загрузи файл.")
        file_info = ctx.uploaded_files[0]
        object_name = file_info.get("object_name") or file_info.get("file_id")
        filename = file_info.get("filename") or file_info.get("file_name", "unknown")
        mime_type = file_info.get("mime_type", "application/octet-stream")
    else:
        # Случай 2: file_ref указан - ищем файл по object_name или file_id
        file_info = next(
            (f for f in ctx.uploaded_files
             if f.get("object_name") == file_ref or f.get("file_id") == file_ref
             or f.get("filename") == file_ref or f.get("file_name") == file_ref),
            None
        )
        if file_info:
            # Найден файл в uploaded_files - используем его object_name
            object_name = file_info.get("object_name") or file_info.get("file_id")
            filename = file_info.get("filename") or file_info.get("file_name", "unknown")
            mime_type = file_info.get("mime_type", "application/octet-stream")
        else:
            # Файл не найден в uploaded_files - используем file_ref как есть (может быть полный путь)
            object_name = file_ref
            filename = file_ref.split("/")[-1] if "/" in file_ref else file_ref
            mime_type = "application/octet-stream"

    if not object_name:
        raise ValueError("Не удалось определить object_name файла")

    # Если MIME-тип неизвестен, определяем по расширению
    if mime_type == "application/octet-stream":
        ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ""
        mime_map = {
            'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'png': 'image/png', 'gif': 'image/gif',
            'bmp': 'image/bmp', 'webp': 'image/webp',
            'mp4': 'video/mp4', 'avi': 'video/x-msvideo',
            'mov': 'video/quicktime', 'mkv': 'video/x-matroska',
            'mp3': 'audio/mpeg', 'wav': 'audio/wav',
            'ogg': 'audio/ogg', 'flac': 'audio/flac',
        }
        mime_type = mime_map.get(ext, "application/octet-stream")

    logger.info(f"Attempting to download file with object_name={object_name}")
    file_object = await minio_service.download_file(object_name)
    if file_object is None:
        raise ValueError(f"Файл {object_name} не найден в хранилище")
    file_object.seek(0)

    size = len(file_object.getbuffer())
    logger.info(f"Loaded file: {filename} ({size} bytes), mime_type={mime_type}, object_name={object_name}")

    return file_object, filename, mime_type


# ============================================================================
# TOOLS
# ============================================================================

@tool
async def analyze_image(
    file_ref: Optional[str] = None,
    prompt: str = "Опиши изображение максимально полезно для владельца питомца.",
    temperature: float = 0.2,
) -> str:
    """Анализировать изображение через GigaChat Vision.
    
    Используй для:
    - Анализа фото питомца (внешний вид, состояние)
    - Оценки симптомов по фото (сыпь, раны, изменения)
    - Идентификации породы
    - Анализа условий содержания
    
    Args:
        file_ref: Ссылка на файл (object_name). Если None - берёт первый загруженный файл
        prompt: Промпт для анализа (что именно нужно описать/найти)
        temperature: Температура генерации (0.0-2.0)
    
    Returns:
        JSON с результатом анализа:
        {
          "analyzed_at": ISO8601,
          "filename": str,
          "prompt": str,
          "analysis": str (текст анализа от GigaChat Vision),
          "file_ref": str
        }
    """
    try:
        ctx = _get_context()

        # Получаем файл с MIME-типом
        file_object, filename, mime_type = await _get_file_from_ref(file_ref)

        # Анализируем через GigaChat Vision
        analysis = await gigachat_client.vision_analysis(
            file=file_object,
            filename=filename,
            prompt=prompt,
            temperature=temperature,
            mime_type=mime_type
        )
        
        result = {
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "prompt": prompt,
            "analysis": analysis,
            "file_ref": file_ref or "auto_selected"
        }
        
        logger.info(f"Image analyzed: {filename}")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to analyze image: {e}")
        return json.dumps({
            "error": str(e),
            "filename": filename if 'filename' in locals() else "unknown"
        }, ensure_ascii=False)


@tool
async def ocr_image(
    file_ref: Optional[str] = None,
    mode: Literal["plain", "structured"] = "structured",
) -> str:
    """Извлечь текст с изображения (OCR).
    
    Используй для:
    - Чтения этикеток кормов
    - Извлечения текста из медицинских справок
    - Распознавания рецептов
    - Чтения результатов анализов
    
    Args:
        file_ref: Ссылка на файл. Если None - берёт первый загруженный
        mode: Режим извлечения:
              - "plain": просто весь текст
              - "structured": пытается структурировать (заголовки, списки)
    
    Returns:
        JSON с распознанным текстом:
        {
          "analyzed_at": ISO8601,
          "filename": str,
          "mode": str,
          "text": str (распознанный текст),
          "structured_data": dict|null (если mode="structured"),
          "file_ref": str
        }
    """
    try:
        ctx = _get_context()
        
        # Получаем файл
        file_object, filename, mime_type = await _get_file_from_ref(file_ref)
        
        # Формируем промпт в зависимости от режима
        if mode == "plain":
            prompt = "Извлеки весь текст с изображения. Верни только текст, без пояснений."
        else:  # structured
            prompt = """Извлеки текст с изображения и структурируй его.
            
Если это этикетка корма:
- Название продукта
- Производитель
- Состав (список ингредиентов)
- Гарантированный анализ (белки, жиры, клетчатка)
- Калорийность
- Дата изготовления/срок годности

Если это медицинский документ:
- Тип документа
- Дата
- Диагноз/показатели
- Рекомендации

Верни в формате JSON с соответствующими полями."""
        
        # OCR через GigaChat Vision
        ocr_result = await gigachat_client.vision_analysis(
            file=file_object,
            filename=filename,
            prompt=prompt,
            temperature=0.1  # Низкая температура для точности
        )
        
        # Пытаемся распарсить JSON если structured mode
        structured_data = None
        if mode == "structured":
            try:
                # Ищем JSON в ответе
                import re
                json_match = re.search(r'\{.*\}', ocr_result, re.DOTALL)
                if json_match:
                    structured_data = json.loads(json_match.group(0))
            except:
                logger.warning("Failed to parse structured OCR result as JSON")
        
        result = {
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "mode": mode,
            "text": ocr_result,
            "structured_data": structured_data,
            "file_ref": file_ref or "auto_selected"
        }
        
        logger.info(f"OCR completed: {filename}, mode={mode}")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to OCR image: {e}")
        return json.dumps({
            "error": str(e),
            "filename": filename if 'filename' in locals() else "unknown"
        }, ensure_ascii=False)


@tool
async def transcribe_audio(
    file_ref: Optional[str] = None,
    audio_format_hint: str = "audio/x-pcm;bit=16;rate=16000",
) -> str:
    """Транскрибировать аудио в текст через SaluteSpeech STT.
    
    Используй для:
    - Транскрипции голосовых сообщений от пользователя
    - Извлечения текста из видео (после извлечения аудио)
    - Создания текстовых записей консультаций ветеринара
    
    Args:
        file_ref: Ссылка на аудио файл. Если None - берёт первый загруженный
        audio_format_hint: Формат аудио (для правильной обработки)
                          Примеры: "audio/x-pcm;bit=16;rate=16000"
                                   "audio/wav"
    
    Returns:
        JSON с транскрипцией:
        {
          "transcribed_at": ISO8601,
          "filename": str,
          "audio_format": str,
          "text": str (распознанный текст),
          "duration_seconds": float|null,
          "file_ref": str
        }
    """
    try:
        ctx = _get_context()
        
        # Получаем файл
        file_object, filename, mime_type = await _get_file_from_ref(file_ref)
        audio_data = file_object.read()
        
        # Определяем параметры аудио из формата
        # Парсим audio/x-pcm;bit=16;rate=16000
        sample_rate = 16000
        bit_depth = 16
        
        if "rate=" in audio_format_hint:
            import re
            rate_match = re.search(r'rate=(\d+)', audio_format_hint)
            if rate_match:
                sample_rate = int(rate_match.group(1))
        
        if "bit=" in audio_format_hint:
            import re
            bit_match = re.search(r'bit=(\d+)', audio_format_hint)
            if bit_match:
                bit_depth = int(bit_match.group(1))
        
        # Транскрибируем через SaluteSpeech
        transcribed_text = await salutespeech_service.speech_to_text(
            audio_data=audio_data,
            sample_rate=sample_rate,
            bit_depth=bit_depth
        )
        
        # Пытаемся определить длительность
        duration = None
        try:
            # Для PCM: duration = bytes / (sample_rate * channels * bytes_per_sample)
            bytes_per_sample = bit_depth // 8
            channels = 1  # Моно по умолчанию
            duration = len(audio_data) / (sample_rate * channels * bytes_per_sample)
        except:
            pass
        
        result = {
            "transcribed_at": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "audio_format": audio_format_hint,
            "text": transcribed_text,
            "duration_seconds": round(duration, 2) if duration else None,
            "file_ref": file_ref or "auto_selected"
        }
        
        logger.info(f"Audio transcribed: {filename}, length={len(transcribed_text)} chars")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        return json.dumps({
            "error": str(e),
            "filename": filename if 'filename' in locals() else "unknown"
        }, ensure_ascii=False)


@tool
async def analyze_video(
    file_ref: Optional[str] = None,
    prompt: str = "Опиши поведение/симптомы на видео. Отдельно: что настораживает и какие следующие шаги.",
    frame_count: int = 10,
    transcribe: bool = True,
) -> str:
    """Анализировать видео (извлечение кадров + анализ через Vision).
    
    Используй для:
    - Анализа поведения питомца на видео
    - Оценки симптомов в динамике
    - Анализа походки, движений
    - Документирования состояния питомца
    
    Args:
        file_ref: Ссылка на видео файл. Если None - берёт первый загруженный
        prompt: Что анализировать на видео
        frame_count: Количество кадров для извлечения (по умолчанию 10)
        transcribe: Извлечь и транскрибировать аудио из видео (по умолчанию True)
    
    Returns:
        JSON с анализом видео:
        {
          "analyzed_at": ISO8601,
          "filename": str,
          "prompt": str,
          "frame_count": int,
          "video_analysis": str (анализ кадров),
          "audio_transcription": str|null (если transcribe=True),
          "frames_analyzed": [
            {"frame_number": int, "timestamp_sec": float}
          ],
          "file_ref": str
        }
    """
    try:
        ctx = _get_context()
        
        # Получаем файл
        file_object, filename, mime_type = await _get_file_from_ref(file_ref)
        video_data = file_object.read()
        
        # Сохраняем временно для обработки
        import tempfile
        import cv2
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video.write(video_data)
            temp_video_path = temp_video.name
        
        try:
            # Извлекаем кадры через OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            
            if not cap.isOpened():
                raise ValueError("Не удалось открыть видео файл")
            
            # Получаем инфо о видео
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Определяем какие кадры извлекать (равномерно по видео)
            frame_count = max(1, frame_count)
            if total_frames <= 0:
                frame_indices = [0]
            else:
                step = max(1, total_frames // frame_count)
                frame_indices = [min(total_frames - 1, i * step) for i in range(frame_count)]
                frame_indices = sorted(set(frame_indices))            
                        # Извлекаем кадры
            extracted_frames = []
            frames_data = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = io.BytesIO(buffer.tobytes())
                    
                    timestamp = idx / fps if fps > 0 else 0
                    
                    extracted_frames.append({
                        "frame_number": idx,
                        "timestamp_sec": round(timestamp, 2)
                    })
                    frames_data.append((frame_bytes, f"frame_{idx}.jpg", "image/jpeg"))
            
            cap.release()
            
            logger.info(f"Extracted {len(frames_data)} frames from video: {filename}")
            
            # Анализируем кадры через GigaChat Vision (multiple images)
            analysis_prompt = f"""Проанализируй ПОДРОБНО эти {len(frames_data)} кадров из видео.

{prompt}

Видео длительностью {duration:.1f} сек. Кадры взяты равномерно на временных отметках:
{chr(10).join([f"• Кадр {i+1}: {extracted_frames[i]['timestamp_sec']:.1f} сек" for i in range(len(extracted_frames))])}

ВАЖНО:
1. Опиши КАЖДЫЙ кадр отдельно - что на нём видно
2. Укажи что МЕНЯЕТСЯ между кадрами (движения, действия, объекты)
3. Опиши общую динамику и развитие событий
4. Обрати внимание на детали: людей, объекты, текст на экране, презентации
5. Если видна презентация/экран - опиши содержимое слайдов

Дай максимально подробное и структурированное описание."""
            
            video_analysis = await gigachat_client.vision_analysis_multiple(
                files=frames_data,
                prompt=analysis_prompt,
                temperature=0.5
            )
            
            # Транскрипция аудио если нужно
            audio_transcription = None
            if transcribe:
                try:
                    # Извлекаем аудио через ffmpeg
                    import subprocess
                    import shutil

                    ffmpeg_path = None

                    try:
                        import imageio_ffmpeg
                        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                        logger.info(f"Using imageio-ffmpeg: {ffmpeg_path}")
                    except ImportError:
                        logger.debug("imageio-ffmpeg not installed, trying system ffmpeg")

                    # Если imageio-ffmpeg нет, ищем системный ffmpeg
                    if not ffmpeg_path:
                        ffmpeg_path = shutil.which('ffmpeg')
                        if ffmpeg_path:
                            logger.info(f"Using system ffmpeg: {ffmpeg_path}")

                    if not ffmpeg_path:
                        logger.warning("ffmpeg not found in PATH and imageio-ffmpeg not installed")
                        audio_transcription = "Транскрипция недоступна: ffmpeg не установлен"
                    else:
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                            temp_audio_path = temp_audio.name

                        # Конвертируем в PCM WAV 16kHz mono
                        result = subprocess.run([
                            ffmpeg_path, '-i', temp_video_path,
                            '-vn',  # Без видео
                            '-acodec', 'pcm_s16le',  # PCM 16-bit
                            '-ar', '16000',  # 16kHz
                            '-ac', '1',  # Моно
                            temp_audio_path,
                            '-y'  # Перезаписать
                        ], capture_output=True, text=True)

                        if result.returncode != 0:
                            logger.warning(f"ffmpeg failed: {result.stderr}")
                            audio_transcription = "Транскрипция недоступна: не удалось извлечь аудио из видео (возможно, видео без звука)."
                        else:
                            # Читаем аудио
                            with open(temp_audio_path, 'rb') as f:
                                audio_data = f.read()

                            # Проверяем что файл не пустой
                            if len(audio_data) < 100:
                                logger.warning("Audio file too small, video might have no audio track")
                                audio_transcription = "Транскрипция недоступна: в видео отсутствует аудиодорожка."
                            else:
                                # Транскрибируем
                                audio_transcription = await salutespeech_service.speech_to_text(
                                    audio_data=audio_data,
                                    sample_rate=16000,
                                    bit_depth=16
                                )

                                logger.info(f"Video audio transcribed: {len(audio_transcription)} chars")

                            # Удаляем временный аудио файл
                            import os
                            try:
                                os.unlink(temp_audio_path)
                            except:
                                pass

                except FileNotFoundError as e:
                    logger.warning(f"ffmpeg not found: {e}")
                    audio_transcription = "Транскрипция недоступна: ffmpeg не установлен. Установите: pip install imageio-ffmpeg"
                except Exception as e:
                    logger.warning(f"Failed to transcribe video audio: {e}")
                    audio_transcription = f"Ошибка транскрипции: {str(e)}"
            
            result = {
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "filename": filename,
                "prompt": prompt,
                "frame_count": len(extracted_frames),
                "video_duration_sec": round(duration, 2),
                "video_analysis": video_analysis,
                "audio_transcription": audio_transcription,
                "frames_analyzed": extracted_frames,
                "file_ref": file_ref or "auto_selected"
            }
            
            logger.info(f"Video analyzed: {filename}, {len(extracted_frames)} frames")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        finally:
            # Удаляем временный видео файл
            import os
            os.unlink(temp_video_path)
        
    except Exception as e:
        logger.error(f"Failed to analyze video: {e}")
        return json.dumps({
            "error": str(e),
            "filename": filename if 'filename' in locals() else "unknown"
        }, ensure_ascii=False)


# ============================================================================
# MULTIMODAL ANALYSIS AGENT
# ============================================================================

class MultimodalAgent:
    """Агент для мультимодального анализа (изображения, видео, аудио)
    
    Возможности:
    - Анализ изображений (GigaChat Vision)
    - OCR текста с изображений
    - Транскрипция аудио (SaluteSpeech STT)
    - Анализ видео (кадры + аудио)
    """
    
    def __init__(self, minio_service: MinioService, llm=None):
        """
        Args:
            minio_service: Сервис для работы с файлами
            llm: LLM для агента
        """
        from app.integrations.gigachat_client import GigaChatClient
        
        self.minio_service = minio_service or MinioServiceDep
        self.llm = llm or GigaChatClient().llm
        
        # Список инструментов
        self.tools = [
            analyze_image,
            ocr_image,
            transcribe_audio,
            analyze_video,
        ]
        
        logger.info("MultimodalAgent initialized with 4 tools")
    
    async def process(
        self,
        user_id: int,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Обработать запрос пользователя"""
        context = context or {}
        token = None
        minio_token = None
        
        try:
            tool_context = MultimodalContext(
                user_id=user_id,
                uploaded_files=context.get("uploaded_files", [])
            )
            
            ctx_token = _multimodal_context.set(tool_context)
            minio_token = _minio_service.set(self.minio_service) 
            
            # Информация о загруженных файлах
            files_info = ""
            if tool_context.uploaded_files:
                files_list = [
                    f"{f.get('filename', 'unknown')} ({f.get('file_type', 'unknown')})"
                    for f in tool_context.uploaded_files[:3]
                ]
                files_info = f"\n📎 Загруженные файлы: {', '.join(files_list)}"
                if len(tool_context.uploaded_files) > 3:
                    files_info += f" и ещё {len(tool_context.uploaded_files) - 3}"
            
            # System prompt
            system_prompt = f"""Ты - эксперт по мультимодальному анализу для владельцев домашних животных.

Пользователь ID: {user_id}{files_info}

**Доступные инструменты (4):**

1. **analyze_image** - Анализ изображений через GigaChat Vision
   Используй для: фото питомца, симптомов, условий содержания, идентификации породы
   
2. **ocr_image** - Извлечение текста с изображений (OCR)
   Используй для: этикеток кормов, справок, результатов анализов, рецептов
   Режимы: "plain" (просто текст) или "structured" (структурированные данные)
   
3. **transcribe_audio** - Транскрипция аудио в текст
   Используй для: голосовых сообщений, аудио из видео
   
4. **analyze_video** - Анализ видео (кадры + опционально аудио)
   Используй для: поведения питомца, симптомов в динамике, походки
   Параметры: frame_count (сколько кадров), transcribe (извлечь аудио)

**file_ref:**
- НЕ УКАЗЫВАЙ file_ref если загружен только ОДИН файл - он возьмется автоматически!
- Указывай file_ref только если загружено НЕСКОЛЬКО файлов и нужно выбрать конкретный
- При указании используй полное значение object_name или file_id из списка загруженных файлов

**Когда использовать какой инструмент:**

ФОТО питомца (file_ref НЕ указываем если файл один!):
→ analyze_image(prompt="Оцени состояние питомца, обрати внимание на...")

ЭТИКЕТКА корма:
→ ocr_image(mode="structured") для извлечения состава

СПРАВКА от ветеринара:
→ ocr_image(mode="structured") для извлечения диагноза, рекомендаций

ГОЛОСОВОЕ сообщение:
→ transcribe_audio()

ВИДЕО с питомцем:
→ analyze_video(prompt="Опиши поведение, обрати внимание на...", frame_count=10, transcribe=True)
  • frame_count - количество кадров (больше = детальнее, но дольше). Для коротких видео (< 1 мин) хватит 5-8, для длинных (> 5 мин) используй 10-15
  • transcribe - извлечь и распознать аудио (True если на видео есть речь/звук)

**Важно:**
- Все инструменты возвращают JSON для оркестратора
- Если файл не подходит для инструмента - скажи об этом
- При медицинских вопросах - рекомендуй консультацию ветеринара
- Будь точным и полезным

Анализируй мультимодальные данные профессионально!"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=settings.DEBUG,
                handle_parsing_errors=True,
                max_iterations=5,
            )
            
            result = await agent_executor.ainvoke({"input": user_message})
            return result.get("output", '{"error": "No output"}')
            
        except Exception as e:
            logger.exception(f"MultimodalAgent error for user {user_id}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        finally:
            if ctx_token is not None:
                _multimodal_context.reset(ctx_token)
            if minio_token is not None:
                _minio_service.reset(minio_token)
