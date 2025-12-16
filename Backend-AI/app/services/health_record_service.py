from typing import Optional, List
from datetime import date
from app.repositories import HealthRecordRepository, PetRepository
from app.dto import HealthRecordCreateDTO, HealthRecordResponseDTO, HealthRecordUpdateDTO
from app.utils.exceptions import HealthRecordNotFoundException,PetNotFoundException,AuthorizationException, ValidationException
from app.models import HealthRecord, RecordType


class HealthRecordService:
    def __init__(self, health_repo: HealthRecordRepository,pet_repository: PetRepository):
        self.health_repo = health_repo
        self.pet_repository = pet_repository
    
    async def _verify_pet_ownership(self, pet_id: int, user_id: int) -> None:
        pet = await self.pet_repository.get_by_id(pet_id, include_deleted=False)
        
        if not pet:
            raise PetNotFoundException(pet_id)
        
        if pet.user_id != user_id:
            raise AuthorizationException()
    
    async def add_health_record(self, user_id: int, record_dto: HealthRecordCreateDTO) -> HealthRecordResponseDTO:
        await self._verify_pet_ownership(record_dto.pet_id, user_id)

        data = record_dto.model_dump(mode="json", exclude_unset=True)
        record = HealthRecord(**data)

        record = await self.health_repo.create(record)
        return HealthRecordResponseDTO.model_validate(record)
    
    async def get_health_record_by_id(self, record_id: int, user_id: int) -> Optional[HealthRecordResponseDTO]:
        record = await self.health_repo.get_by_id(record_id, include_deleted=False)
        
        if not record:
            return None
        
        await self._verify_pet_ownership(record.pet_id, user_id)
        
        return HealthRecordResponseDTO.model_validate(record)
    
    async def get_pet_health_records(self, pet_id: int, user_id: int,include_deleted: bool = False) -> List[HealthRecordResponseDTO]:
        await self._verify_pet_ownership(pet_id, user_id)
        
        records = await self.health_repo.get_by_pet_id(
            pet_id, 
            include_deleted=include_deleted
        )
        
        return [HealthRecordResponseDTO.model_validate(record) for record in records]
    
    async def get_records_by_type(self,pet_id: int,user_id: int,record_type: str,) -> List[HealthRecordResponseDTO]:
        await self._verify_pet_ownership(pet_id, user_id)

        try:
            record_type_enum = RecordType(record_type)
        except ValueError:
            raise ValidationException(f"Неизвестный тип записи: {record_type}")

        records = await self.health_repo.get_by_type(
            pet_id=pet_id,
            record_type=record_type_enum,
        )

        return [HealthRecordResponseDTO.model_validate(record) for record in records]
    
    async def get_unresolved_records(self,pet_id: int,user_id: int) -> List[HealthRecordResponseDTO]:
        await self._verify_pet_ownership(pet_id, user_id)
        
        records = await self.health_repo.get_unresolved(pet_id)
        
        return [HealthRecordResponseDTO.model_validate(record) for record in records]
    
    async def update_health_record(self, record_id: int, user_id: int, record_dto: HealthRecordUpdateDTO) -> HealthRecordResponseDTO:
        record = await self.health_repo.get_by_id(record_id, include_deleted=False)
        if not record:
            raise HealthRecordNotFoundException(record_id)

        await self._verify_pet_ownership(record.pet_id, user_id)

        data = record_dto.model_dump(mode="json", exclude_unset=True)
        if not data:
            return HealthRecordResponseDTO.model_validate(record)

        for field, value in data.items():
            setattr(record, field, value)

        record = await self.health_repo.update(record)
        return HealthRecordResponseDTO.model_validate(record)

    
    async def soft_delete_health_record(self, record_id: int, user_id: int) -> bool:
        record = await self.health_repo.get_by_id(record_id, include_deleted=False)
        if not record:
            raise HealthRecordNotFoundException(record_id)

        await self._verify_pet_ownership(record.pet_id, user_id)

        record.soft_delete()
        await self.health_repo.update(record)
        return True

    async def restore_health_record(self, record_id: int, user_id: int) -> HealthRecordResponseDTO:
        record = await self.health_repo.get_by_id(record_id, include_deleted=True)
        if not record:
            raise HealthRecordNotFoundException(record_id)

        await self._verify_pet_ownership(record.pet_id, user_id)

        record.restore()
        record = await self.health_repo.update(record)
        return HealthRecordResponseDTO.model_validate(record)
