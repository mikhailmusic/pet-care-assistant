from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.dto import UserResponseDTO
from app.utils.security import decode_access_token
from app.utils.exceptions import InvalidTokenException, TokenExpiredException
from .services import UserServiceDep


security = HTTPBearer(auto_error=True)


async def get_current_user(
    user_service: UserServiceDep,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> UserResponseDTO:
    try:
        token = credentials.credentials
        token_payload = decode_access_token(token)
    except (InvalidTokenException, TokenExpiredException) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await user_service.get_user_by_id(token_payload.user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь удален",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_superuser(
    current_user: UserResponseDTO = Depends(get_current_user),
) -> UserResponseDTO:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав. Требуются права администратора.",
        )
    return current_user


CurrentUser = Annotated[UserResponseDTO, Depends(get_current_user)]
CurrentSuperuser = Annotated[UserResponseDTO, Depends(get_current_superuser)]