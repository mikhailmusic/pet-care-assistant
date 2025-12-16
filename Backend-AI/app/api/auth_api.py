from fastapi import APIRouter, status

from app.dto import UserCreateDTO, UserLoginDTO, UserResponseDTO, TokenResponseDTO, UserUpdateDTO
from app.utils.security import create_access_token
from app.dependencies import CurrentUser
from app.dependencies import UserServiceDep


router = APIRouter()


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
