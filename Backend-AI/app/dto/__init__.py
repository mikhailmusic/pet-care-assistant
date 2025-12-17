from .common import (
    TimestampDTO,
    PaginationParams,
    PaginatedResponse,
)


from .user_dto import (
    UserCreateDTO,
    UserUpdateDTO,
    UserResponseDTO,
    UserProfileDTO,
    TokenPayloadDTO,
    UserLoginDTO,
    ChangePasswordDTO,
    TokenResponseDTO,
    UserWithCredsDTO
)
from .google_auth_dto import (
    GoogleAuthUrlResponseDTO,
    GoogleAuthCodeDTO,
    GoogleCredentialsDTO,
)

from .chat_dto import (
    ChatCreateDTO,
    ChatUpdateDTO,
    ChatSettingsDTO,
    ChatResponseDTO,
    ChatListItemDTO,
)

from .message_dto import (
    MessageCreateDTO,
    MessageUpdateDTO,
    MessageResponseDTO,
    # SendMessageDTO,
    # MessageWithFilesDTO,
    StreamMessageChunkDTO,
)

from .pet_dto import (
    PetCreateDTO,
    PetUpdateDTO,
    PetResponseDTO,
    PetListItemDTO,
)

from .health_record_dto import (
    HealthRecordCreateDTO,
    HealthRecordUpdateDTO,
    HealthRecordResponseDTO,
    HealthRecordListItemDTO,
)

from .calendar_event_dto import (
    CalendarEventCreateDTO,
    CalendarEventUpdateDTO,
    CalendarEventResponseDTO,
)

from .file_dto import (
    FileUploadResponseDTO,
    MultipleFileUploadResponseDTO,
    FileMetadataDTO,
)


__all__ = [
    # Common
    "BaseDTO",
    "TimestampDTO",
    "PaginationParams",
    "PaginatedResponse",
    "MessageResponse",
    
    # Auth
    "RegisterDTO",
    "LoginDTO",
    "TokenDTO",
    "TokenDataDTO",
    "ChangePasswordDTO",
    
    # User
    "UserCreateDTO",
    "UserUpdateDTO",
    "UserResponseDTO",
    "UserProfileDTO",
    
    # Chat
    "ChatCreateDTO",
    "ChatUpdateDTO",
    "ChatSettingsDTO",
    "ChatResponseDTO",
    "ChatListItemDTO",
    
    # Message
    "MessageCreateDTO",
    "MessageUpdateDTO",
    "MessageResponseDTO",
    "SendMessageDTO",
    "MessageWithFilesDTO",
    "StreamMessageChunkDTO",
    
    # Pet
    "PetCreateDTO",
    "PetUpdateDTO",
    "PetResponseDTO",
    "PetListItemDTO",
    "ExtractedPetInfoDTO",
    
    # Health Record
    "HealthRecordCreateDTO",
    "HealthRecordUpdateDTO",
    "HealthRecordResponseDTO",
    "HealthRecordListItemDTO",
    
    # Calendar Event
    "CalendarEventCreateDTO",
    "CalendarEventUpdateDTO",
    "CalendarEventResponseDTO",
    "CalendarEventListItemDTO",
    "UpcomingEventsDTO",
    
    # File
    "FileUploadResponseDTO",
    "MultipleFileUploadResponseDTO",
    "FileMetadataDTO",
    "DeleteFileDTO",
    
    # Email
    "EmailSendDTO",
    "EmailResponseDTO",
    "SendReportEmailDTO",
    
    # RAG
    "DocumentUploadDTO",
    "DocumentChunkDTO",
    "RAGSearchDTO",
    "RAGSearchResultDTO",
    "RAGSearchResponseDTO",
    "DocumentListItemDTO",
    "DeleteDocumentDTO",
]
