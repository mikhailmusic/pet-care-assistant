from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from .common import TimestampDTO


class EventTypeDTO(str, Enum):
    vaccination = "vaccination"
    vet_visit = "vet_visit"
    grooming = "grooming"
    medication = "medication"
    feeding = "feeding"
    exercise = "exercise"
    other = "other"


class CalendarEventCreateDTO(BaseModel):
    pet_id: Optional[int] = None
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    event_type: EventTypeDTO
    
    start_datetime: datetime
    end_datetime: Optional[datetime] = None
    timezone: str = Field(default="Europe/Moscow", max_length=64)
    
    location: Optional[str] = Field(None, max_length=500)
    
    is_recurring: bool = False
    recurrence_rule: Optional[str] = Field(None, max_length=500)


class CalendarEventUpdateDTO(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    event_type: Optional[EventTypeDTO] = None
    
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    timezone: Optional[str] = Field(None, max_length=64)
    
    location: Optional[str] = Field(None, max_length=500)
    
    is_recurring: Optional[bool] = None
    recurrence_rule: Optional[str] = Field(None, max_length=500)


class CalendarEventResponseDTO(TimestampDTO):
    user_id: int
    pet_id: Optional[int] = None
    
    title: str
    description: Optional[str] = None
    event_type: EventTypeDTO
    
    start_datetime: datetime
    end_datetime: Optional[datetime] = None
    timezone: str
    
    location: Optional[str] = None
    
    google_event_id: Optional[str] = None
    google_calendar_id: str
    
    is_recurring: bool
    recurrence_rule: Optional[str] = None


