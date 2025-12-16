from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from .common import TimestampDTO


class UserCreateDTO(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=64)
    full_name: str = Field(..., max_length=255)


class UserUpdateDTO(BaseModel):
    full_name: str = Field(..., max_length=255)

class UserLoginDTO(BaseModel):
    email: EmailStr
    password: str


class ChangePasswordDTO(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=8, max_length=64)




class UserResponseDTO(TimestampDTO):
    email: str
    full_name: str
    is_superuser: bool

class UserWithCredsDTO(UserResponseDTO):
    google_credentials_json: Optional[str] = None

class UserProfileDTO(TimestampDTO):
    email: str
    full_name: str
    is_superuser: bool

    total_chats: int = 0
    total_pets: int = 0
    total_messages: int = 0


class TokenResponseDTO(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponseDTO

class TokenPayloadDTO(BaseModel):
    user_id: int
    email: str
    exp: Optional[int] = None