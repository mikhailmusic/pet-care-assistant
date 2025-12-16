from typing import Optional, List
from app.repositories import PetRepository
from app.dto import PetCreateDTO, PetResponseDTO, PetUpdateDTO
from app.utils.exceptions import PetNotFoundException, AuthorizationException
from app.models import Pet


class PetService:
    def __init__(self, repository: PetRepository):
        self.repository = repository
    
    async def add_pet(self, user_id: int, pet_dto: PetCreateDTO) -> PetResponseDTO:
        data = pet_dto.model_dump(exclude_unset=True)
        pet = Pet(user_id=user_id, **data)
        pet = await self.repository.create(pet)
        return PetResponseDTO.model_validate(pet)
    
    async def get_pet_by_id(self, pet_id: int, user_id: Optional[int] = None) -> Optional[PetResponseDTO]:
        pet = await self.repository.get_by_id(pet_id, include_deleted=False)
        if not pet:
            return None
        if user_id is not None and pet.user_id != user_id:
            raise AuthorizationException()
        return PetResponseDTO.model_validate(pet)

    async def get_user_pets(self, user_id: int) -> List[PetResponseDTO]:
        pets = await self.repository.get_by_user_id(user_id)
        return [PetResponseDTO.model_validate(pet) for pet in pets]

    async def update_pet(self, pet_id: int, user_id: int, pet_dto: PetUpdateDTO) -> PetResponseDTO:
        pet = await self.repository.get_by_id(pet_id, include_deleted=False)
        if not pet:
            raise PetNotFoundException(pet_id)
        if pet.user_id != user_id:
            raise AuthorizationException()

        data = pet_dto.model_dump(exclude_unset=True)
        for field, value in data.items():
            setattr(pet, field, value)

        pet = await self.repository.update(pet)
        return PetResponseDTO.model_validate(pet)

    async def soft_delete_pet(self, pet_id: int, user_id: int) -> bool:
        pet = await self.repository.get_by_id(pet_id, include_deleted=False)
        if not pet:
            raise PetNotFoundException(pet_id)
        if pet.user_id != user_id:
            raise AuthorizationException()

        pet.soft_delete()
        await self.repository.update(pet)
        return True

    async def restore_pet(self, pet_id: int, user_id: int) -> PetResponseDTO:
        pet = await self.repository.get_by_id(pet_id, include_deleted=True)
        if not pet:
            raise PetNotFoundException(pet_id)
        if pet.user_id != user_id:
            raise AuthorizationException()

        pet.restore()
        pet = await self.repository.update(pet)
        return PetResponseDTO.model_validate(pet)
