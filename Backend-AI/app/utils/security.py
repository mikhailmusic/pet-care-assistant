from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from passlib.context import CryptContext
from loguru import logger

from app.config import settings
from app.dto import TokenPayloadDTO
from app.utils.exceptions import InvalidTokenException, TokenExpiredException


pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int, email: str, expires_delta: Optional[timedelta] = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES))

    to_encode = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": now,
        "type": "access",
    }
    
    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> TokenPayloadDTO:
    try:
        if settings.DEBUG:
            logger.info(f"Decoding JWT with key: {settings.JWT_SECRET_KEY[:8]}..., alg={settings.JWT_ALGORITHM}")

        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        
        user_id = payload.get("sub")
        email = payload.get("email")
        
        if user_id is None or email is None:
            raise InvalidTokenException()
        
        return TokenPayloadDTO(
            user_id=int(user_id), 
            email=email,
            exp=payload.get("exp"),
        )
        
    except ExpiredSignatureError:
        raise TokenExpiredException()
    except JWTError:
        raise InvalidTokenException()
