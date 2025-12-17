from typing import Optional
from app.repositories import UserRepository
from app.dto import UserCreateDTO, UserResponseDTO, UserUpdateDTO, UserWithCredsDTO
from app.utils.security import hash_password, verify_password
from app.utils.exceptions import EmailAlreadyExistsException,UserNotFoundException, InvalidCredentialsException
from app.models import User


class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    async def add_user(self, user_dto: UserCreateDTO) -> UserResponseDTO:
        if await self.repository.email_exists(user_dto.email):
            raise EmailAlreadyExistsException()

        hashed_password = hash_password(user_dto.password)

        user = User(
            email=user_dto.email,
            password_hash=hashed_password,
            full_name=user_dto.full_name,
            is_superuser=False,
            google_credentials_json=user_dto.google_credentials_json,
        )
        user = await self.repository.create(user)
        return UserResponseDTO.model_validate(user)

    async def get_user_by_id(self, user_id: int) -> Optional[UserResponseDTO]:
        user = await self.repository.get_by_id(user_id, include_deleted=False)
        if not user:
            return None
        return UserResponseDTO.model_validate(user)

    async def get_user_by_email(self, email: str) -> Optional[UserResponseDTO]:
        user = await self.repository.get_by_email(email, include_deleted=False)
        if not user:
            return None
        return UserResponseDTO.model_validate(user)

    async def change_user(self, user_id: int, user_dto: UserUpdateDTO) -> UserResponseDTO:
        user = await self.repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundException(user_id)

        data = user_dto.model_dump(exclude_unset=True)
        if data:
            for field, value in data.items():
                setattr(user, field, value)
            user = await self.repository.update(user)

        return UserResponseDTO.model_validate(user)

    async def authenticate(self, email: str, password: str) -> UserResponseDTO:
        user = await self.repository.get_by_email(email, include_deleted=False)
        if not user or not verify_password(password, user.password_hash):
            raise InvalidCredentialsException()
        return UserResponseDTO.model_validate(user)

    async def soft_delete_user(self, user_id: int) -> bool:
        user = await self.repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundException(user_id)

        user.soft_delete()
        await self.repository.update(user)
        return True

    async def restore_user(self, user_id: int) -> UserResponseDTO:
        user = await self.repository.get_by_id(user_id, include_deleted=True)
        if not user:
            raise UserNotFoundException(user_id)

        user.restore()
        user = await self.repository.update(user)
        return UserResponseDTO.model_validate(user)

    async def add_google_credentials(self, user_id: int, creds_json: str) -> UserWithCredsDTO:
        user = await self.repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundException(user_id)

        user.google_credentials_json = creds_json
        user = await self.repository.update(user)
        return UserWithCredsDTO.model_validate(user)

    async def get_google_credentials(self, user_id: int) -> Optional[str]:
        user = await self.repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundException(user_id)
        return user.google_credentials_json
