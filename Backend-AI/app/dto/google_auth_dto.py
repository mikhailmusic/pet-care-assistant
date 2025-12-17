from typing import Optional
from pydantic import BaseModel, AnyUrl, Field


class GoogleAuthUrlResponseDTO(BaseModel):
    auth_url: AnyUrl
    state: Optional[str] = None


class GoogleAuthCodeDTO(BaseModel):
    code: str = Field(..., min_length=4)
    redirect_uri: AnyUrl


class GoogleCredentialsDTO(BaseModel):
    google_credentials_json: str
