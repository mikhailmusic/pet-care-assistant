from enum import Enum

from typing import Optional, List
from datetime import date
from pydantic import BaseModel, Field
from .common import TimestampDTO
from .file_dto import FileMetadataDTO

class RecordTypeDTO(str, Enum):
    VACCINATION = "vaccination"
    EXAMINATION = "examination"
    TREATMENT = "treatment"
    SURGERY = "surgery"
    ANALYSIS = "analysis"
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    WEIGHT = "weight"
    BEHAVIOR = "behavior"
    OTHER = "other"

class UrgencyLevelDTO(str, Enum):
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class HealthRecordCreateDTO(BaseModel):
    pet_id: int
    record_type: RecordTypeDTO
    record_date: date
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    
    symptoms: Optional[str] = None
    diagnosis: Optional[str] = Field(None, max_length=255)
    treatment: Optional[str] = None
    medications_prescribed: Optional[str] = None
    
    weight_kg: Optional[float] = Field(None, gt=0)
    temperature_c: Optional[float] = Field(None, ge=35, le=45)
    
    urgency: UrgencyLevelDTO
    is_resolved: bool    
    
    vet_name: Optional[str] = Field(None, max_length=255)
    vet_clinic: Optional[str] = Field(None, max_length=255)
    
    cost: Optional[float] = Field(None, ge=0)
    next_visit_date: Optional[date] = None


class HealthRecordUpdateDTO(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    
    symptoms: Optional[str] = None
    diagnosis: Optional[str] = Field(None, max_length=255)
    treatment: Optional[str] = None
    medications_prescribed: Optional[str] = None
    
    weight_kg: Optional[float] = Field(None, gt=0)
    temperature_c: Optional[float] = Field(None, ge=35, le=45)
    
    urgency: Optional[UrgencyLevelDTO] = None
    is_resolved: Optional[bool]
    
    vet_name: Optional[str] = Field(None, max_length=255)
    vet_clinic: Optional[str] = Field(None, max_length=255)
    files: Optional[List[FileMetadataDTO]] = None
    metadata_json: Optional[dict] = None
    
    cost: Optional[float] = Field(None, ge=0)
    next_visit_date: Optional[date] = None


class HealthRecordResponseDTO(TimestampDTO):    
    pet_id: int
    record_type: RecordTypeDTO
    record_date: date
    title: str
    description: Optional[str] = None
    symptoms: Optional[str] = None
    diagnosis: Optional[str] = None
    treatment: Optional[str] = None
    medications_prescribed: Optional[str] = None
    weight_kg: Optional[float] = None
    temperature_c: Optional[float] = None
    urgency: UrgencyLevelDTO
    is_resolved: bool
    vet_name: Optional[str] = None
    vet_clinic: Optional[str] = None
    metadata_json: Optional[dict] = None

    files: Optional[List[FileMetadataDTO]] = None
    cost: Optional[float] = None
    next_visit_date: Optional[date] = None
    

class HealthRecordListItemDTO(TimestampDTO):    
    pet_id: int
    record_type: RecordTypeDTO
    record_date: date
    title: str
    
    urgency: UrgencyLevelDTO
    is_resolved: bool
    
    vet_clinic: Optional[str] = None