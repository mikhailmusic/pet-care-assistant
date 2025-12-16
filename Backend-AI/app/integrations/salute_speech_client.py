import uuid
import httpx
from datetime import datetime, timedelta, timezone
from typing import Optional, BinaryIO
from loguru import logger

from app.config import settings
from app.utils.exceptions import SaluteSpeechException


class SaluteSpeechClient:

    OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    STT_URL = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
    TTS_URL = "https://smartspeech.sber.ru/rest/v1/text:synthesize"

    def __init__(self):
        self.auth_token = settings.SALUTESPEECH_API_KEY
        self.scope = "SALUTE_SPEECH_PERS"
        
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        
        logger.info("SaluteSpeechService initialized")


    async def _get_access_token(self, force_refresh: bool = False) -> str:
        if not force_refresh and self._access_token and self._token_expires_at:
            if datetime.now(timezone.utc) < self._token_expires_at - timedelta(minutes=5):
                logger.debug("Using cached SaluteSpeech access token")
                return self._access_token
        
        rq_uid = str(uuid.uuid4())
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "RqUID": rq_uid,
            "Authorization": f"Basic {self.auth_token}"
        }
        
        payload = {"scope": self.scope}

        try:
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.post(
                    self.OAUTH_URL,
                    headers=headers,
                    data=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                data = response.json()
                self._access_token = data["access_token"]
                
                self._token_expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
                
                logger.info(f"SaluteSpeech access token obtained, expires at {self._token_expires_at}")
                return self._access_token
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get SaluteSpeech token: {e.response.status_code} {e.response.text}")
            raise SaluteSpeechException(f"Ошибка получения токена: {e.response.status_code}")
        except Exception as e:
            logger.error(f"SaluteSpeech token error: {e}")
            raise SaluteSpeechException(f"Ошибка получения токена: {e}")
        


    async def speech_to_text(self,audio_data: bytes,sample_rate: int = 16000,bit_depth: int = 16) -> str:

        try:
            token = await self._get_access_token()
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": f"audio/x-pcm;bit={bit_depth};rate={sample_rate}"
            }
            
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.post(
                    self.STT_URL,
                    headers=headers,
                    content=audio_data,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    recognized_text = result.get("result", "")
                    
                    logger.info(f"Speech recognized: {recognized_text[:50]}...")
                    return recognized_text
                    
                elif response.status_code == 401:
                    logger.warning("Token expired, refreshing...")
                    token = await self._get_access_token(force_refresh=True)
                    
                    headers["Authorization"] = f"Bearer {token}"
                    response = await client.post(
                        self.STT_URL,
                        headers=headers,
                        content=audio_data,
                        timeout=60.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return result.get("result", "")
                    else:
                        raise SaluteSpeechException(
                            f"Ошибка распознавания речи: {response.status_code} {response.text}"
                        )
                else:
                    raise SaluteSpeechException(
                        f"Ошибка распознавания речи: {response.status_code} {response.text}"
                    )
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"SaluteSpeech STT error: {e.response.status_code} {e.response.text}")
            raise SaluteSpeechException(f"Ошибка STT: {e.response.status_code}")
        except SaluteSpeechException:
            raise
        except Exception as e:
            logger.error(f"SaluteSpeech STT error: {e}")
            raise SaluteSpeechException(f"Ошибка распознавания речи: {e}")



    async def text_to_speech(self,text: str, voice: str = "Bys_24000", format: str = "wav16",) -> bytes:
        try:
            token = await self._get_access_token()
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/text"
            }
            
            params = {
                "format": format,
                "voice": voice
            }
            
            async with httpx.AsyncClient(verify=False) as client:
                response = await client.post(
                    self.TTS_URL,
                    headers=headers,
                    params=params,
                    content=text.encode("utf-8"),
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    audio_bytes = response.content
                    
                    logger.info(
                        f"Speech synthesized: {len(text)} chars -> {len(audio_bytes)} bytes, "
                        f"voice={voice}, format={format}"
                    )
                    return audio_bytes
                    
                elif response.status_code == 401:
                    logger.warning("Token expired, refreshing...")
                    token = await self._get_access_token(force_refresh=True)
                    
                    headers["Authorization"] = f"Bearer {token}"
                    response = await client.post(
                        self.TTS_URL,
                        headers=headers,
                        params=params,
                        content=text.encode("utf-8"),
                        timeout=60.0
                    )
                    
                    if response.status_code == 200:
                        return response.content
                    else:
                        raise SaluteSpeechException(
                            f"Ошибка синтеза речи: {response.status_code} {response.text}"
                        )
                else:
                    raise SaluteSpeechException(
                        f"Ошибка синтеза речи: {response.status_code} {response.text}"
                    )
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"SaluteSpeech TTS error: {e.response.status_code} {e.response.text}")
            raise SaluteSpeechException(f"Ошибка TTS: {e.response.status_code}")
        except SaluteSpeechException:
            raise
        except Exception as e:
            logger.error(f"SaluteSpeech TTS error: {e}")
            raise SaluteSpeechException(f"Ошибка синтеза речи: {e}")



salutespeech_service = SaluteSpeechClient()
