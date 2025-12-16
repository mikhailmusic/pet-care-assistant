from .services import (
    UserServiceDep,
    ChatServiceDep,
    PetServiceDep,
    MessageServiceDep,
    FileServiceDep
)
from .auth import (
    CurrentUser,
    CurrentSuperuser,
)

__all__ = [
    # сервисы
    "UserServiceDep",
    "ChatServiceDep",
    "PetServiceDep",
    "MessageServiceDep",
    "FileServiceDep"
    # пользователи
    "CurrentUser",
    "CurrentSuperuser",
]
