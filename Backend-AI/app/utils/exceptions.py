from fastapi import HTTPException, status


class PetCareException(Exception):
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class AuthenticationException(PetCareException):   
    def __init__(self, message: str = "Ошибка аутентификации"):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED)


class InvalidCredentialsException(AuthenticationException):
    def __init__(self):
        super().__init__("Неверный email или пароль")


class TokenExpiredException(AuthenticationException):    
    def __init__(self):
        super().__init__("Токен истек, войдите заново")


class InvalidTokenException(AuthenticationException):  
    def __init__(self):
        super().__init__("Невалидный токен")



class AuthorizationException(PetCareException): 
    def __init__(self, message: str = "Недостаточно прав"):
        super().__init__(message, status.HTTP_403_FORBIDDEN)



class NotFoundException(PetCareException):
    def __init__(self, message: str = "Ресурс не найден"):
        super().__init__(message, status.HTTP_404_NOT_FOUND)


class UserNotFoundException(NotFoundException):
    def __init__(self, user_id: int | None = None):
        message = f"Пользователь с ID {user_id} не найден" if user_id else "Пользователь не найден"
        super().__init__(message)


class ChatNotFoundException(NotFoundException): 
    def __init__(self, chat_id: int | None = None):
        message = f"Чат с ID {chat_id} не найден" if chat_id else "Чат не найден"
        super().__init__(message)


class PetNotFoundException(NotFoundException):
    def __init__(self, pet_id: int | None = None):
        message = f"Питомец с ID {pet_id} не найден" if pet_id else "Питомец не найден"
        super().__init__(message)

class HealthRecordNotFoundException(NotFoundException):
    def __init__(self, record_id: int | None = None):
        message = f"Медицинская запись с ID {record_id} не найдена" if record_id else "Медицинская запись не найдена"
        super().__init__(message)

class MessageNotFoundException(NotFoundException):
    def __init__(self, message_id: int | None = None):
        message = f"Сообщение с ID {message_id} не найдено" if message_id else "Сообщение не найдено"
        super().__init__(message)


class FileNotFoundException(NotFoundException):
    def __init__(self, filename: str | None = None):
        message = f"Файл '{filename}' не найден" if filename else "Файл не найден"
        super().__init__(message)



class ValidationException(PetCareException):
    def __init__(self, message: str = "Ошибка валидации данных"):
        super().__init__(message, status.HTTP_422_UNPROCESSABLE_ENTITY)


class InvalidFileTypeException(ValidationException):
    def __init__(self, allowed_types: list[str] | None = None):
        if allowed_types:
            message = f"Недопустимый тип файла. Разрешены: {', '.join(allowed_types)}"
        else:
            message = "Недопустимый тип файла"
        super().__init__(message)


class FileTooLargeException(ValidationException):
    def __init__(self, max_size_mb: int):
        super().__init__(f"Файл слишком большой. Максимальный размер: {max_size_mb}MB")


class EmailAlreadyExistsException(ValidationException):
    def __init__(self):
        super().__init__("Пользователь с таким email уже существует")



class BusinessLogicException(PetCareException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)


class ChatLimitExceededException(BusinessLogicException):
    def __init__(self, max_chats: int = 50):
        super().__init__(f"Достигнут максимальный лимит чатов ({max_chats})")


class MessageTooLongException(BusinessLogicException):
    def __init__(self, max_length: int = 10000):
        super().__init__(f"Сообщение слишком длинное. Максимум: {max_length} символов")



class ExternalServiceException(PetCareException):
    def __init__(self, service_name: str, message: str = "Ошибка внешнего сервиса"):
        super().__init__(f"{service_name}: {message}", status.HTTP_503_SERVICE_UNAVAILABLE)


class GigaChatException(ExternalServiceException):
    def __init__(self, message: str = "Ошибка при обращении к GigaChat"):
        super().__init__("GigaChat", message)


class SaluteSpeechException(ExternalServiceException):
    def __init__(self, message: str = "Ошибка при обращении к SaluteSpeech"):
        super().__init__("SaluteSpeech", message)


class MinIOException(ExternalServiceException):
    def __init__(self, message: str = "Ошибка при работе с файловым хранилищем"):
        super().__init__("MinIO", message)


class EmailSendException(ExternalServiceException):
    def __init__(self, message: str = "Ошибка при отправке email"):
        super().__init__("SMTP", message)


class DatabaseException(PetCareException):
    def __init__(self, message: str = "Ошибка базы данных"):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)


class GoogleCalendarException(ExternalServiceException):
    def __init__(self, message: str = "Ошибка при работе с Google Calendar"):
        self.message = message
        super().__init__(self.message)

def convert_to_http_exception(exc: PetCareException) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail=exc.message,
    )