from typing import Optional, Dict, Any
from datetime import date
from pydantic import BaseModel, Field
from .common import TimestampDTO

class PetCreateDTO(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    species: str = Field(..., min_length=1, max_length=50)
    breed: Optional[str] = Field(None, max_length=100)

    gender: Optional[str] = Field(None, max_length=20)
    color: Optional[str] = Field(None, max_length=100)
    birth_date: Optional[date] = None
    age_years: Optional[int] = Field(None, ge=0, le=100)
    age_months: Optional[int] = Field(None, ge=0, le=11)

    weight_kg: Optional[float] = Field(None, gt=0, le=1000)
    height_cm: Optional[float] = Field(None, gt=0, le=500)
    is_sterilized: Optional[bool] = None
    microchip_number: Optional[str] = Field(None, max_length=50)

    allergies: Optional[str] = None
    chronic_conditions: Optional[str] = None
    medications: Optional[str] = None

    diet_type: Optional[str] = Field(None, max_length=50)
    activity_level: Optional[str] = Field(None, max_length=50)
    profile_image_url: Optional[str] = Field(None, max_length=500)


class PetUpdateDTO(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    species: Optional[str] = Field(None, min_length=1, max_length=50)
    breed: Optional[str] = Field(None, max_length=100)
    
    gender: Optional[str] = Field(None, max_length=20)
    color: Optional[str] = Field(None, max_length=100)
    birth_date: Optional[date] = None
    age_years: Optional[int] = Field(None, ge=0, le=100)
    age_months: Optional[int] = Field(None, ge=0, le=11)
    
    weight_kg: Optional[float] = Field(None, gt=0, le=1000)
    height_cm: Optional[float] = Field(None, gt=0, le=500)
    is_sterilized: Optional[bool] = None
    microchip_number: Optional[str] = Field(None, max_length=50)
    
    allergies: Optional[str] = None
    chronic_conditions: Optional[str] = None
    medications: Optional[str] = None
    
    diet_type: Optional[str] = Field(None, max_length=50)
    activity_level: Optional[str] = Field(None, max_length=50)
    metadata_json: Optional[Dict[str, Any]] = None
    
    profile_image_url: Optional[str] = Field(None, max_length=500)


class PetResponseDTO(TimestampDTO):   
    user_id: int
    name: str
    species: str
    breed: Optional[str] = None
    
    gender: Optional[str] = None
    color: Optional[str] = None
    birth_date: Optional[date] = None
    age_years: Optional[int] = None
    age_months: Optional[int] = None
    
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    is_sterilized: Optional[bool] = None
    microchip_number: Optional[str] = None
    
    allergies: Optional[str] = None
    chronic_conditions: Optional[str] = None
    medications: Optional[str] = None
    
    diet_type: Optional[str] = None
    activity_level: Optional[str] = None
    
    metadata_json: Optional[Dict[str, Any]] = None
    profile_image_url: Optional[str] = None


class PetDetailDTO(PetResponseDTO):
    health_records_count: int = 0
    upcoming_events_count: int = 0
    last_health_record_date: Optional[date] = None


class PetListItemDTO(TimestampDTO):
    user_id: int
    name: str
    species: str
    breed: Optional[str] = None
    
    age_years: Optional[int] = None
    profile_image_url: Optional[str] = None
    
    health_records_count: int = 0
    last_health_check_date: Optional[date] = None
