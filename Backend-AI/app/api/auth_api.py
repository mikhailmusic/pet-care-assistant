from fastapi import APIRouter, status, HTTPException

from app.dto import (
    UserCreateDTO,
    UserLoginDTO,
    UserResponseDTO,
    TokenResponseDTO,
    UserUpdateDTO,
    GoogleAuthUrlResponseDTO,
    GoogleAuthCodeDTO,
    GoogleCredentialsDTO,
)
from app.utils.security import create_access_token
from app.dependencies import CurrentUser
from app.dependencies import UserServiceDep
from app.integrations import google_calendar_service
from app.utils.exceptions import GoogleCalendarException


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/google/url", response_model=GoogleAuthUrlResponseDTO)
async def get_google_auth_url(redirect_uri: str, state: str | None = None):
    """
    Возвращает URL авторизации Google OAuth2 для получения access/refresh токенов Calendar.
    """
    try:
        auth_url, resolved_state = google_calendar_service.get_authorization_url(
            redirect_uri=redirect_uri,
            state=state,
        )
        return GoogleAuthUrlResponseDTO(auth_url=auth_url, state=resolved_state)
    except GoogleCalendarException as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.post("/google/exchange", response_model=GoogleCredentialsDTO)
async def exchange_google_code(payload: GoogleAuthCodeDTO):
    """
    Обменивает code из redirect на JSON учетных данных Google (access/refresh tokens).
    Возвращенный `google_credentials_json` нужно сохранить при регистрации пользователя.
    """
    try:
        creds_json = google_calendar_service.exchange_code_for_credentials(
            code=payload.code,
            redirect_uri=str(payload.redirect_uri),
        )
        return GoogleCredentialsDTO(google_credentials_json=creds_json)
    except GoogleCalendarException as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.post("/register", response_model=TokenResponseDTO, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreateDTO,
    service: UserServiceDep,
):
    user = await service.add_user(user_data)
    access_token = create_access_token(user_id=user.id, email=user.email)

    return TokenResponseDTO(
        access_token=access_token,
        token_type="bearer",
        user=user,
    )


@router.post("/login", response_model=TokenResponseDTO)
async def login(
    credentials: UserLoginDTO,
    service: UserServiceDep,
):
    user_dto = await service.authenticate(
        email=credentials.email,
        password=credentials.password,
    )
    access_token = create_access_token(user_id=user_dto.id, email=user_dto.email)

    return TokenResponseDTO(
        access_token=access_token,
        token_type="bearer",
        user=user_dto,
    )


@router.get("/me", response_model=UserResponseDTO)
async def get_current_user_info(
    current_user: CurrentUser,
):
    return current_user


@router.patch("/me", response_model=UserResponseDTO)
async def update_current_user(
    user_data: UserUpdateDTO,
    current_user: CurrentUser,
    service: UserServiceDep,
):
    updated_user = await service.change_user(
        user_id=current_user.id,
        user_dto=user_data,
    )
    return updated_user
