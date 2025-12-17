from __future__ import annotations

from typing import Any, Dict, List, Optional, BinaryIO, AsyncIterator
from langchain_gigachat import GigaChat
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import re
import asyncio
from loguru import logger

from app.config import settings
from app.utils.exceptions import GigaChatException


class GigaChatClient:
    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None) -> None:
        # Initialize a base client with env defaults; per-request overrides are applied via .bind
        default_model = model or settings.GIGACHAT_MODEL
        default_temp = settings.GIGACHAT_TEMPERATURE if temperature is None else temperature

        base_params = dict(
            credentials=settings.GIGACHAT_API_KEY,
            scope=settings.GIGACHAT_SCOPE,
            model=default_model,
            verify_ssl_certs=settings.GIGACHAT_VERIFY_SSL_CERTS,
            temperature=default_temp,
        )

        self.default_model = default_model
        self.llm = GigaChat(**base_params)
        self.llm_stream = GigaChat(**base_params, streaming=True)

        logger.info(f"GigaChatClient initialized: model={default_model}, temperature={default_temp}")

    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[SystemMessage | HumanMessage | AIMessage]:
        langchain_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                logger.warning(f"Unknown message role: {role}, treating as user")
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages

    async def chat_completion(
        self,messages: List[Dict[str, str]],temperature: Optional[float] = None,max_tokens: Optional[int] = None,model: Optional[str] = None,) -> str:
        try:
            langchain_messages = self._convert_messages(messages)

            llm = self.llm

            if temperature is not None:
                llm = llm.bind(temperature=temperature)
            if max_tokens is not None:
                llm = llm.bind(max_tokens=max_tokens)
            if model is not None:
                llm = llm.bind(model=model)

            response = await llm.ainvoke(langchain_messages)
            return response.content

        except Exception as e:
            logger.error(f"GigaChat completion error: {e}")
            raise GigaChatException(f"Ошибка генерации текста: {e}")

    async def chat_completion_stream(
        self,messages: List[Dict[str, str]],temperature: Optional[float] = None,max_tokens: Optional[int] = None, model: Optional[str] = None,) -> AsyncIterator[str]:
        try:
            langchain_messages = self._convert_messages(messages)

            llm = self.llm_stream

            if temperature is not None:
                llm = llm.bind(temperature=temperature)
            if max_tokens is not None:
                llm = llm.bind(max_tokens=max_tokens)
            if model is not None:
                llm = llm.bind(model=model)

            async for chunk in llm.astream(langchain_messages):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"GigaChat streaming error: {e}")
            raise GigaChatException(f"Ошибка streaming генерации: {e}")
        

    async def vision_analysis(self,file: BinaryIO,filename: str,prompt: str,temperature: Optional[float] = None,) -> str:
        try:
            file.seek(0)
            uploaded_file = await asyncio.to_thread(self.llm.upload_file, file)
            file_id = uploaded_file.id_
            logger.info(f"File '{filename}' uploaded to GigaChat with ID: {file_id}")

            # 2. Формируем сообщение с прикрепленным файлом
            message = HumanMessage(
                content=prompt,
                additional_kwargs={"attachments": [file_id]},
            )

            llm = self.llm
            if temperature is not None:
                llm = llm.bind(temperature=temperature)

            response = await llm.ainvoke([message])

            logger.info(f"Vision analysis completed for file_id: {file_id}")
            return response.content

        except Exception as e:
            logger.error(f"GigaChat vision error: {e}")
            raise GigaChatException(f"Ошибка анализа изображения: {e}")

    async def vision_analysis_multiple(
        self, files: List[tuple[BinaryIO, str]],  # [(file, filename), ...]
        prompt: str, temperature: Optional[float] = None, ) -> str:
        try:
            file_ids = []
            for file, filename in files:
                file.seek(0)
                uploaded_file = await asyncio.to_thread(self.llm.upload_file, file)
                file_ids.append(uploaded_file.id_)
                logger.info(f"Uploaded: {filename} -> {uploaded_file.id_}")
            
            message = HumanMessage(
                content=prompt,
                additional_kwargs={"attachments": file_ids},
            )
            
            llm = self.llm
            if temperature is not None:
                llm = llm.bind(temperature=temperature)
            
            response = await llm.ainvoke([message])
            
            logger.info(f"Vision analysis completed for {len(file_ids)} files")
            return response.content
            
        except Exception as e:
            logger.error(f"GigaChat multiple vision error: {e}")
            raise GigaChatException(f"Ошибка анализа изображений: {e}")

    async def generate_image(self,prompt: str,width: int = 1024,height: int = 1024,) -> str:
        try:
            # LLM с поддержкой function calling для генерации изображений
            llm_with_functions = self.llm.bind(function_call="auto")

            message = HumanMessage(
                content=f"Сгенерируй изображение: {prompt}. Размер: {width}x{height}"
            )

            response = await llm_with_functions.ainvoke([message])
            content = response.content

            # Извлечение URL/ID изображения из ответа
            img_match = re.search(r'<img\s+src="([^"]+)"', content)
            
            if img_match:
                file_id = img_match.group(1)
                logger.info(f"Image generated with file_id: {file_id}")
                return file_id

            logger.warning("Image ID not found in content, returning raw content: " f"{content[:200]}...")    
            return content

        except Exception as e:
            logger.error(f"GigaChat image generation error: {e}")
            raise GigaChatException(f"Ошибка генерации изображения: {e}")

    async def download_file(self, file_id: str) -> bytes:
        try:
            import base64
            
            # Получаем изображение через приватный API клиента
            img = await asyncio.to_thread(self.llm._client.get_image, file_id)
            
            # Декодируем из base64
            img_bytes = base64.b64decode(img.content)
            
            logger.info(f"Downloaded and decoded file: {file_id}, size: {len(img_bytes)} bytes")
            return img_bytes
            
        except Exception as e:
            logger.error(f"GigaChat file download error: {e}")
            raise GigaChatException(f"Ошибка скачивания файла: {e}")


gigachat_client = GigaChatClient()


def create_llm_from_settings(chat_settings: Optional[Dict[str, Any]] = None):
    """
    Build a LangChain GigaChat LLM instance using chat settings stored in DB.
    Falls back to env defaults when settings are not provided.
    """
    settings = chat_settings or {}
    model = settings.get("gigachat_model") or settings.get("model")
    temperature = settings.get("temperature")
    max_tokens = settings.get("max_tokens")

    client = GigaChatClient(model=model, temperature=temperature)
    llm = client.llm

    bind_params: Dict[str, Any] = {}
    if model:
        bind_params["model"] = model
    if temperature is not None:
        bind_params["temperature"] = temperature
    if max_tokens is not None:
        bind_params["max_tokens"] = max_tokens

    if bind_params:
        return llm.bind(**bind_params)

    return llm
