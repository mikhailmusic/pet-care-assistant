from __future__ import annotations

from typing import Optional, Annotated
from datetime import date
from loguru import logger

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from app.services.pet_service import PetService
from app.services.health_record_service import HealthRecordService
from app.dto import PetCreateDTO, PetUpdateDTO, PetResponseDTO, HealthRecordCreateDTO, HealthRecordUpdateDTO, HealthRecordResponseDTO

from app.models.health_record import RecordType, UrgencyLevel


def _format_pet_full(pet: PetResponseDTO) -> str:
    """Форматировать полную информацию о питомце"""
    lines = [
        f"🐾 **{pet.name}** (ID: {pet.id})",
        f"Вид: {pet.species}",
    ]
    
    if pet.breed:
        lines.append(f"Порода: {pet.breed}")
    if pet.gender:
        lines.append(f"Пол: {pet.gender}")
    if pet.color:
        lines.append(f"Окрас: {pet.color}")
    
    if pet.birth_date:
        lines.append(f"Дата рождения: {pet.birth_date.strftime('%d.%m.%Y')}")
    if pet.age_years is not None:
        age_str = f"{pet.age_years} лет"
        if pet.age_months:
            age_str += f" {pet.age_months} мес"
        lines.append(f"Возраст: {age_str}")
    
    if pet.weight_kg:
        lines.append(f"Вес: {pet.weight_kg} кг")
    if pet.height_cm:
        lines.append(f"Рост: {pet.height_cm} см")
    
    if pet.is_sterilized is not None:
        lines.append(f"Стерилизован: {'Да' if pet.is_sterilized else 'Нет'}")
    if pet.microchip_number:
        lines.append(f"Микрочип: {pet.microchip_number}")
    if pet.allergies:
        lines.append(f"Аллергии: {pet.allergies}")
    if pet.chronic_conditions:
        lines.append(f"Хронические заболевания: {pet.chronic_conditions}")
    if pet.medications:
        lines.append(f"Принимаемые лекарства: {pet.medications}")
    
    if pet.diet_type:
        lines.append(f"Тип питания: {pet.diet_type}")
    if pet.activity_level:
        lines.append(f"Уровень активности: {pet.activity_level}")
    
    return "\n".join(lines)


def _format_health_record_full(record: HealthRecordResponseDTO) -> str:
    """Форматировать полную информацию о медицинской записи"""
    lines = [
        f"📋 **{record.title}** (ID: {record.id})",
        f"Тип: {record.record_type.value}",
        f"Дата: {record.record_date.strftime('%d.%m.%Y')}",
        f"Срочность: {record.urgency.value}",
        f"Статус: {'Решено ✅' if record.is_resolved else 'Не решено ⏳'}",
    ]
    
    if record.description:
        lines.append(f"Описание: {record.description}")
    if record.symptoms:
        lines.append(f"Симптомы: {record.symptoms}")
    if record.diagnosis:
        lines.append(f"Диагноз: {record.diagnosis}")
    if record.treatment:
        lines.append(f"Лечение: {record.treatment}")
    if record.medications_prescribed:
        lines.append(f"Назначенные лекарства: {record.medications_prescribed}")
    
    if record.weight_kg:
        lines.append(f"Вес на момент записи: {record.weight_kg} кг")
    if record.temperature_c:
        lines.append(f"Температура: {record.temperature_c}°C")
    
    if record.vet_name:
        lines.append(f"Ветеринар: {record.vet_name}")
    if record.vet_clinic:
        lines.append(f"Клиника: {record.vet_clinic}")
    
    if record.cost:
        lines.append(f"Стоимость: {record.cost} руб.")
    if record.next_visit_date:
        lines.append(f"Следующий визит: {record.next_visit_date.strftime('%d.%m.%Y')}")
    
    return "\n".join(lines)



class PetMemoryTools:
    
    def __init__(self, pet_service: PetService, health_service: HealthRecordService, ):
        self.pet_service = pet_service
        self.health_service = health_service
    

    @tool
    async def create_pet_profile(
        self,
        state: Annotated[dict, InjectedState],
        name: str,
        species: str,
        breed: Optional[str] = None,
        gender: Optional[str] = None,
        color: Optional[str] = None,
        birth_date: Optional[str] = None,
        age_years: Optional[int] = None,
        age_months: Optional[int] = None,
        weight_kg: Optional[float] = None,
        height_cm: Optional[float] = None,
        is_sterilized: Optional[bool] = None,
        microchip_number: Optional[str] = None,
        allergies: Optional[str] = None,
        chronic_conditions: Optional[str] = None,
        medications: Optional[str] = None,
        diet_type: Optional[str] = None,
        activity_level: Optional[str] = None,
    ) -> str:
        """Создать профиль нового питомца.
        
        Используй когда пользователь впервые упоминает питомца.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            name: Имя питомца (обязательно)
            species: Вид - собака/кошка/попугай и т.д. (обязательно)
            breed: Порода
            gender: Пол (самец/самка)
            color: Окрас
            birth_date: Дата рождения в формате YYYY-MM-DD
            age_years: Возраст в годах
            age_months: Дополнительные месяцы
            weight_kg: Вес в кг
            height_cm: Рост/высота в см
            is_sterilized: Стерилизован (true/false)
            microchip_number: Номер микрочипа
            allergies: Аллергии
            chronic_conditions: Хронические заболевания
            medications: Принимаемые лекарства
            diet_type: Тип питания
            activity_level: Уровень активности
        
        Returns:
            Результат создания профиля
        """
        try:
            user_id = state["user_id"]
            
            # Проверка на дубликат
            user_pets = await self.pet_service.get_user_pets(user_id)
            if any(p.name.lower() == name.lower() for p in user_pets):
                return f"❌ Питомец '{name}' уже существует. Используй update_pet_profile для обновления."
            
            # Парсинг даты
            parsed_birth_date = None
            if birth_date:
                try:
                    parsed_birth_date = date.fromisoformat(birth_date)
                except ValueError:
                    return f"❌ Неверный формат даты: {birth_date}. Используй YYYY-MM-DD"
            
            # Собираем данные
            pet_data = {
                "name": name,
                "species": species,
                "breed": breed,
                "gender": gender,
                "color": color,
                "birth_date": parsed_birth_date,
                "age_years": age_years,
                "age_months": age_months,
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "is_sterilized": is_sterilized,
                "microchip_number": microchip_number,
                "allergies": allergies,
                "chronic_conditions": chronic_conditions,
                "medications": medications,
                "diet_type": diet_type,
                "activity_level": activity_level,
            }
            
            pet_data = {k: v for k, v in pet_data.items() if v is not None}
            
            create_dto = PetCreateDTO(**pet_data)
            new_pet = await self.pet_service.add_pet(user_id=user_id, pet_dto=create_dto)
            
            logger.info(f"Created pet: {name} (ID: {new_pet.id}) for user {user_id}")
            return f"✅ Создан профиль питомца:\n\n{_format_pet_full(new_pet)}"
            
        except Exception as e:
            logger.error(f"Failed to create pet: {e}")
            return f"❌ Ошибка создания профиля: {str(e)}"
    
    @tool
    async def update_pet_profile(
        self,
        state: Annotated[dict, InjectedState],
        pet_name: str,
        breed: Optional[str] = None,
        gender: Optional[str] = None,
        color: Optional[str] = None,
        birth_date: Optional[str] = None,
        age_years: Optional[int] = None,
        age_months: Optional[int] = None,
        weight_kg: Optional[float] = None,
        height_cm: Optional[float] = None,
        is_sterilized: Optional[bool] = None,
        microchip_number: Optional[str] = None,
        allergies: Optional[str] = None,
        chronic_conditions: Optional[str] = None,
        medications: Optional[str] = None,
        diet_type: Optional[str] = None,
        activity_level: Optional[str] = None,
    ) -> str:
        """Обновить информацию о существующем питомце.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            pet_name: Имя питомца (обязательно)
            (остальные поля опциональны)
        
        Returns:
            Результат обновления
        """
        try:
            user_id = state["user_id"]
            
            user_pets = await self.pet_service.get_user_pets(user_id)
            pet = next((p for p in user_pets if p.name.lower() == pet_name.lower()), None)
            
            if not pet:
                return f"❌ Питомец '{pet_name}' не найден. Используй create_pet_profile."
            
            # Парсинг даты
            parsed_birth_date = None
            if birth_date:
                try:
                    parsed_birth_date = date.fromisoformat(birth_date)
                except ValueError:
                    return f"❌ Неверный формат даты: {birth_date}"
            
            # Собираем данные
            update_data = {
                "breed": breed,
                "gender": gender,
                "color": color,
                "birth_date": parsed_birth_date,
                "age_years": age_years,
                "age_months": age_months,
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "is_sterilized": is_sterilized,
                "microchip_number": microchip_number,
                "allergies": allergies,
                "chronic_conditions": chronic_conditions,
                "medications": medications,
                "diet_type": diet_type,
                "activity_level": activity_level,
            }
            
            update_data = {k: v for k, v in update_data.items() if v is not None}
            
            if not update_data:
                return f"❌ Нет данных для обновления '{pet_name}'"
            
            update_dto = PetUpdateDTO(**update_data)
            updated_pet = await self.pet_service.update_pet(
                pet_id=pet.id,
                user_id=user_id,
                pet_dto=update_dto
            )
            
            logger.info(f"Updated pet: {pet_name} (ID: {pet.id})")
            
            updated_fields = ", ".join(update_data.keys())
            return f"✅ Обновлён профиль '{pet_name}'\nИзменено: {updated_fields}\n\n{_format_pet_full(updated_pet)}"
            
        except Exception as e:
            logger.error(f"Failed to update pet: {e}")
            return f"❌ Ошибка обновления: {str(e)}"
    
    @tool
    async def get_pet_info(
        self,
        state: Annotated[dict, InjectedState],
        pet_name: str
    ) -> str:
        """Получить ПОЛНУЮ информацию о конкретном питомце.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            pet_name: Имя питомца
        
        Returns:
            Полная информация о питомце
        """
        try:
            user_id = state["user_id"]
            
            user_pets = await self.pet_service.get_user_pets(user_id)
            pet = next((p for p in user_pets if p.name.lower() == pet_name.lower()), None)
            
            if not pet:
                available = ", ".join([p.name for p in user_pets])
                return f"❌ Питомец '{pet_name}' не найден. Доступные: {available}"
            
            return _format_pet_full(pet)
            
        except Exception as e:
            logger.error(f"Failed to get pet info: {e}")
            return f"❌ Ошибка получения информации: {str(e)}"
    
    @tool
    async def list_user_pets(
        self,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Получить список ВСЕХ питомцев пользователя с полной информацией.
        
        Используй когда пользователь спрашивает "какие у меня питомцы", "покажи всех питомцев".
        
        Args:
            state: Состояние графа (автоматически инжектится)
        
        Returns:
            Список всех питомцев с полной информацией
        """
        try:
            user_id = state["user_id"]
            
            user_pets = await self.pet_service.get_user_pets(user_id)
            
            if not user_pets:
                return "У вас пока нет зарегистрированных питомцев."
            
            result = [f"📋 Ваши питомцы ({len(user_pets)}):\n"]
            
            for i, pet in enumerate(user_pets, 1):
                result.append(f"\n{'='*50}")
                result.append(f"Питомец #{i}:")
                result.append(_format_pet_full(pet))
            
            return "\n".join(result)
            
        except Exception as e:
            logger.error(f"Failed to list pets: {e}")
            return f"❌ Ошибка получения списка: {str(e)}"
    
    @tool
    async def delete_pet(
        self,
        state: Annotated[dict, InjectedState],
        pet_name: str
    ) -> str:
        """Удалить питомца.
        
        ВНИМАНИЕ: Удаление питомца также удалит все его медицинские записи!
        
        Args:
            state: Состояние графа (автоматически инжектится)
            pet_name: Имя питомца для удаления
        
        Returns:
            Результат удаления
        """
        try:
            user_id = state["user_id"]
            
            user_pets = await self.pet_service.get_user_pets(user_id)
            pet = next((p for p in user_pets if p.name.lower() == pet_name.lower()), None)
            
            if not pet:
                return f"❌ Питомец '{pet_name}' не найден."
            
            await self.pet_service.soft_delete_pet(pet_id=pet.id, user_id=user_id)
            
            logger.info(f"Deleted pet: {pet_name} (ID: {pet.id})")
            return f"✅ Питомец '{pet_name}' удалён (включая все медицинские записи)"
            
        except Exception as e:
            logger.error(f"Failed to delete pet: {e}")
            return f"❌ Ошибка удаления: {str(e)}"
    
    # ========================================================================
    # HEALTH RECORD MANAGEMENT TOOLS
    # ========================================================================
    
    @tool
    async def add_health_record(
        self,
        state: Annotated[dict, InjectedState],
        pet_name: str,
        title: str,
        record_type: str,
        record_date: Optional[str] = None,
        description: Optional[str] = None,
        symptoms: Optional[str] = None,
        diagnosis: Optional[str] = None,
        treatment: Optional[str] = None,
        medications_prescribed: Optional[str] = None,
        urgency: str = "medium",
        is_resolved: bool = False,
        vet_name: Optional[str] = None,
        vet_clinic: Optional[str] = None,
        weight_kg: Optional[float] = None,
        temperature_c: Optional[float] = None,
        cost: Optional[float] = None,
        next_visit_date: Optional[str] = None,
    ) -> str:
        """Добавить медицинскую запись о питомце.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            pet_name: Имя питомца (обязательно)
            title: Название записи (обязательно)
            record_type: vaccination/examination/treatment/surgery/analysis/symptom/medication/weight/behavior/other
            record_date: Дата в формате YYYY-MM-DD (по умолчанию сегодня)
            description: Подробное описание
            symptoms: Симптомы
            diagnosis: Диагноз
            treatment: Лечение
            medications_prescribed: Назначенные лекарства
            urgency: low/medium/high/critical
            is_resolved: Решено (true/false)
            vet_name: Имя ветеринара
            vet_clinic: Клиника
            weight_kg: Вес
            temperature_c: Температура
            cost: Стоимость
            next_visit_date: Дата следующего визита (YYYY-MM-DD)
        
        Returns:
            Результат создания записи
        """
        try:
            user_id = state["user_id"]
            
            # Находим питомца
            user_pets = await self.pet_service.get_user_pets(user_id)
            pet = next((p for p in user_pets if p.name.lower() == pet_name.lower()), None)
            
            if not pet:
                if len(user_pets) == 1:
                    pet = user_pets[0]
                else:
                    available = ", ".join([p.name for p in user_pets])
                    return f"❌ Питомец '{pet_name}' не найден. Доступные: {available}"
            
            # Парсим record_type
            try:
                record_type_enum = RecordType(record_type.lower())
            except ValueError:
                valid = ", ".join([t.value for t in RecordType])
                return f"❌ Неверный тип '{record_type}'. Доступные: {valid}"
            
            # Парсим urgency
            try:
                urgency_enum = UrgencyLevel(urgency.lower())
            except ValueError:
                valid = ", ".join([u.value for u in UrgencyLevel])
                return f"❌ Неверная срочность '{urgency}'. Доступные: {valid}"
            
            # Парсим даты
            record_date_obj = date.today()
            if record_date:
                try:
                    record_date_obj = date.fromisoformat(record_date)
                except ValueError:
                    return f"❌ Неверный формат даты: {record_date}"
            
            next_visit_date_obj = None
            if next_visit_date:
                try:
                    next_visit_date_obj = date.fromisoformat(next_visit_date)
                except ValueError:
                    return f"❌ Неверный формат даты визита: {next_visit_date}"
            
            # Создаём запись
            health_data = {
                "pet_id": pet.id,
                "record_type": record_type_enum,
                "record_date": record_date_obj,
                "title": title,
                "description": description,
                "symptoms": symptoms,
                "diagnosis": diagnosis,
                "treatment": treatment,
                "medications_prescribed": medications_prescribed,
                "urgency": urgency_enum,
                "is_resolved": is_resolved,
                "vet_name": vet_name,
                "vet_clinic": vet_clinic,
                "weight_kg": weight_kg,
                "temperature_c": temperature_c,
                "cost": cost,
                "next_visit_date": next_visit_date_obj,
            }
            
            health_data = {k: v for k, v in health_data.items() if v is not None}
            
            create_dto = HealthRecordCreateDTO(**health_data)
            new_record = await self.health_service.add_health_record(
                user_id=user_id,
                record_dto=create_dto
            )
            
            logger.info(f"Created health record: {title} (ID: {new_record.id}) for {pet.name}")
            return f"✅ Добавлена запись для {pet.name}:\n\n{_format_health_record_full(new_record)}"
            
        except Exception as e:
            logger.error(f"Failed to add health record: {e}")
            return f"❌ Ошибка добавления записи: {str(e)}"
    
    @tool
    async def update_health_record(
        self,
        state: Annotated[dict, InjectedState],
        record_id: int,
        title: Optional[str] = None,
        description: Optional[str] = None,
        symptoms: Optional[str] = None,
        diagnosis: Optional[str] = None,
        treatment: Optional[str] = None,
        medications_prescribed: Optional[str] = None,
        urgency: Optional[str] = None,
        is_resolved: Optional[bool] = None,
        vet_name: Optional[str] = None,
        vet_clinic: Optional[str] = None,
        weight_kg: Optional[float] = None,
        temperature_c: Optional[float] = None,
        cost: Optional[float] = None,
        next_visit_date: Optional[str] = None,
    ) -> str:
        """Обновить существующую медицинскую запись.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            record_id: ID записи (обязательно)
            (остальные поля опциональны)
        
        Returns:
            Результат обновления
        """
        try:
            user_id = state["user_id"]
            
            # Парсим urgency если указан
            urgency_enum = None
            if urgency:
                try:
                    urgency_enum = UrgencyLevel(urgency.lower())
                except ValueError:
                    valid = ", ".join([u.value for u in UrgencyLevel])
                    return f"❌ Неверная срочность '{urgency}'. Доступные: {valid}"
            
            # Парсим дату
            next_visit_date_obj = None
            if next_visit_date:
                try:
                    next_visit_date_obj = date.fromisoformat(next_visit_date)
                except ValueError:
                    return f"❌ Неверный формат даты: {next_visit_date}"
            
            # Собираем данные
            update_data = {
                "title": title,
                "description": description,
                "symptoms": symptoms,
                "diagnosis": diagnosis,
                "treatment": treatment,
                "medications_prescribed": medications_prescribed,
                "urgency": urgency_enum,
                "is_resolved": is_resolved,
                "vet_name": vet_name,
                "vet_clinic": vet_clinic,
                "weight_kg": weight_kg,
                "temperature_c": temperature_c,
                "cost": cost,
                "next_visit_date": next_visit_date_obj,
            }
            
            update_data = {k: v for k, v in update_data.items() if v is not None}
            
            if not update_data:
                return f"❌ Нет данных для обновления записи {record_id}"
            
            update_dto = HealthRecordUpdateDTO(**update_data)
            updated_record = await self.health_service.update_health_record(
                record_id=record_id,
                user_id=user_id,
                record_dto=update_dto
            )
            
            logger.info(f"Updated health record: {record_id}")
            
            updated_fields = ", ".join(update_data.keys())
            return f"✅ Обновлена запись {record_id}\nИзменено: {updated_fields}\n\n{_format_health_record_full(updated_record)}"
            
        except Exception as e:
            logger.error(f"Failed to update health record: {e}")
            return f"❌ Ошибка обновления: {str(e)}"
    
    @tool
    async def get_health_record(
        self,
        state: Annotated[dict, InjectedState],
        record_id: int
    ) -> str:
        """Получить ПОЛНУЮ информацию о медицинской записи.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            record_id: ID записи
        
        Returns:
            Полная информация о записи
        """
        try:
            user_id = state["user_id"]
            
            record = await self.health_service.get_health_record_by_id(
                record_id=record_id,
                user_id=user_id
            )
            
            if not record:
                return f"❌ Запись {record_id} не найдена."
            
            return _format_health_record_full(record)
            
        except Exception as e:
            logger.error(f"Failed to get health record: {e}")
            return f"❌ Ошибка получения записи: {str(e)}"
    
    @tool
    async def list_pet_health_records(
        self,
        state: Annotated[dict, InjectedState],
        pet_name: str,
        limit: int = 10
    ) -> str:
        """Получить список медицинских записей питомца.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            pet_name: Имя питомца
            limit: Максимальное количество записей (по умолчанию 10)
        
        Returns:
            Список медицинских записей
        """
        try:
            user_id = state["user_id"]
            
            # Находим питомца
            user_pets = await self.pet_service.get_user_pets(user_id)
            pet = next((p for p in user_pets if p.name.lower() == pet_name.lower()), None)
            
            if not pet:
                return f"❌ Питомец '{pet_name}' не найден."
            
            # Получаем записи
            records = await self.health_service.get_pet_health_records(
                pet_id=pet.id,
                user_id=user_id
            )
            
            if not records:
                return f"У {pet_name} пока нет медицинских записей."
            
            # Ограничиваем количество
            records = records[:limit]
            
            result = [f"📋 Медицинские записи {pet_name} (показано {len(records)}):\n"]
            
            for i, record in enumerate(records, 1):
                result.append(f"\n{'='*50}")
                result.append(f"Запись #{i}:")
                result.append(_format_health_record_full(record))
            
            return "\n".join(result)
            
        except Exception as e:
            logger.error(f"Failed to list health records: {e}")
            return f"❌ Ошибка получения записей: {str(e)}"
    
    @tool
    async def delete_health_record(
        self,
        state: Annotated[dict, InjectedState],
        record_id: int
    ) -> str:
        """Удалить медицинскую запись.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            record_id: ID записи для удаления
        
        Returns:
            Результат удаления
        """
        try:
            user_id = state["user_id"]
            
            await self.health_service.soft_delete_health_record(
                record_id=record_id,
                user_id=user_id
            )
            
            logger.info(f"Deleted health record: {record_id}")
            return f"✅ Медицинская запись {record_id} удалена"
            
        except Exception as e:
            logger.error(f"Failed to delete health record: {e}")
            return f"❌ Ошибка удаления: {str(e)}"



def create_pet_memory_agent(
    pet_service: PetService,
    health_service: HealthRecordService,
    llm,
    name: str = "pet_memory",
):
    """Создать агента для работы с питомцами и медицинскими записями
    
    Args:
        pet_service: Сервис для работы с питомцами
        health_service: Сервис для работы с медицинскими записями
        llm: Языковая модель
        name: Имя агента (для supervisor handoff)
    
    Returns:
        Compiled ReAct agent
    """
    tools_instance = PetMemoryTools(pet_service, health_service)
    
    # Собираем все методы, помеченные как @tool
    tools = [
        tools_instance.create_pet_profile,
        tools_instance.update_pet_profile,
        tools_instance.get_pet_info,
        tools_instance.list_user_pets,
        tools_instance.delete_pet,
        tools_instance.add_health_record,
        tools_instance.update_health_record,
        tools_instance.get_health_record,
        tools_instance.list_pet_health_records,
        tools_instance.delete_health_record,
    ]
    
    prompt = (
        "Ты - помощник по уходу за домашними животными.\n\n"
        "Ты управляешь данными о питомцах пользователя:\n"
        "- Создаёшь и обновляешь профили питомцев\n"
        "- Ведёшь медицинские записи (прививки, анализы, посещения врача)\n"
        "- Предоставляешь информацию о питомцах по запросу\n\n"
        "Когда пользователь упоминает питомца - автоматически сохраняй информацию.\n"
        "Будь точным и полезным!"
    )
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        name=name,
        prompt=prompt,
    )
    
    logger.info(f"Created PetMemoryAgent '{name}' with {len(tools)} tools")
    return agent