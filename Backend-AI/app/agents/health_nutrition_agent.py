# app/agents/health_nutrition_agent.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime, date, timedelta, timezone
from loguru import logger
from contextvars import ContextVar
import json

from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.services.pet_service import PetService
from app.services.health_record_service import HealthRecordService
from app.integrations.gigachat_client import GigaChatClient
from app.config import settings


# ============================================================================
# CONTEXT
# ============================================================================

@dataclass
class HealthNutritionContext:
    """Контекст для Health & Nutrition Agent"""
    user_id: int
    current_pet_id: Optional[int] = None
    current_pet_name: str = ""


_health_nutrition_context: ContextVar[Optional[HealthNutritionContext]] = ContextVar(
    '_health_nutrition_context',
    default=None
)

_pet_service: ContextVar[Optional[PetService]] = ContextVar('_pet_service', default=None)
_health_service: ContextVar[Optional[HealthRecordService]] = ContextVar('_health_service', default=None)


def _get_context() -> HealthNutritionContext:
    """Get the current context from ContextVar"""
    ctx = _health_nutrition_context.get()
    if ctx is None:
        raise RuntimeError("HealthNutrition context not set.")
    return ctx


def _get_pet_service() -> PetService:
    """Get pet service from ContextVar"""
    service = _pet_service.get()
    if service is None:
        raise RuntimeError("Pet service not set.")
    return service


def _get_health_service() -> HealthRecordService:
    """Get health service from ContextVar"""
    service = _health_service.get()
    if service is None:
        raise RuntimeError("Health service not set.")
    return service


# ============================================================================
# TOOLS
# ============================================================================

@tool
async def analyze_health_records(
    pet_name: str,
    period_days: int = 90,
    unresolved_only: bool = False,
    max_records: int = 50,
) -> str:
    """Анализировать медицинские записи питомца за период.
    
    Собирает и анализирует все медицинские записи: симптомы, диагнозы, лечение,
    прививки, анализы. Выявляет паттерны и тренды.
    
    Args:
        pet_name: Имя питомца
        period_days: Период анализа в днях (по умолчанию 90)
        unresolved_only: Показать только нерешённые проблемы
        max_records: Максимальное количество записей для анализа
    
    Returns:
        JSON с анализом медицинских записей:
        {
          "pet_name": str,
          "period_days": int,
          "analyzed_at": ISO8601,
          "total_records": int,
          "records": [
            {
              "id": int,
              "date": str,
              "type": str,
              "title": str,
              "urgency": str,
              "is_resolved": bool,
              "symptoms": str,
              "diagnosis": str,
              "treatment": str,
              "weight_kg": float,
              "temperature_c": float,
              "vet_clinic": str
            }
          ],
          "statistics": {
            "by_type": {"vaccination": int, "symptom": int, ...},
            "by_urgency": {"critical": int, "high": int, ...},
            "unresolved_count": int,
            "weight_measurements": [{"date": str, "weight_kg": float}],
            "temperature_measurements": [{"date": str, "temperature_c": float}]
          },
          "patterns": {
            "frequent_issues": [str],
            "weight_trend": "increasing|decreasing|stable|insufficient_data",
            "weight_change_kg": float,
            "weight_change_percent": float
          }
        }
    """
    try:
        ctx = _get_context()
        pet_service = _get_pet_service()
        health_service = _get_health_service()
        
        # Находим питомца
        user_pets = await pet_service.get_user_pets(ctx.user_id)
        pet = next((p for p in user_pets if p.name.lower() == pet_name.lower()), None)
        
        if not pet:
            return json.dumps({
                "error": f"Питомец '{pet_name}' не найден",
                "pet_name": pet_name
            }, ensure_ascii=False)
        
        # Получаем медицинские записи за период
        cutoff_date = date.today() - timedelta(days=period_days)
        all_records = await health_service.get_pet_health_records(
            pet_id=pet.id,
            user_id=ctx.user_id
        )
        
        # Фильтруем по дате
        records = [r for r in all_records if r.record_date >= cutoff_date]
        
        # Фильтруем только нерешённые если нужно
        if unresolved_only:
            records = [r for r in records if not r.is_resolved]
        
        # Ограничиваем количество
        records = records[:max_records]
        
        # Сортируем по дате (новые первые)
        records.sort(key=lambda x: x.record_date, reverse=True)
        
        # Форматируем записи для вывода
        formatted_records = []
        for r in records:
            formatted_records.append({
                "id": r.id,
                "date": r.record_date.isoformat(),
                "type": r.record_type.value,
                "title": r.title,
                "urgency": r.urgency.value,
                "is_resolved": r.is_resolved,
                "symptoms": r.symptoms,
                "diagnosis": r.diagnosis,
                "treatment": r.treatment,
                "medications": r.medications_prescribed,
                "weight_kg": r.weight_kg,
                "temperature_c": r.temperature_c,
                "vet_clinic": r.vet_clinic,
                "vet_name": r.vet_name,
                "cost": r.cost,
            })
        
        # Статистика по типам
        by_type = {}
        for r in records:
            record_type = r.record_type.value
            by_type[record_type] = by_type.get(record_type, 0) + 1
        
        # Статистика по срочности
        by_urgency = {
            "critical": sum(1 for r in records if r.urgency.value == "critical"),
            "high": sum(1 for r in records if r.urgency.value == "high"),
            "medium": sum(1 for r in records if r.urgency.value == "medium"),
            "low": sum(1 for r in records if r.urgency.value == "low"),
        }
        
        # Нерешённые
        unresolved_count = sum(1 for r in records if not r.is_resolved)
        
        # Измерения веса
        weight_measurements = [
            {"date": r.record_date.isoformat(), "weight_kg": r.weight_kg}
            for r in records if r.weight_kg is not None
        ]
        weight_measurements.sort(key=lambda x: x["date"])
        
        # Измерения температуры
        temperature_measurements = [
            {"date": r.record_date.isoformat(), "temperature_c": r.temperature_c}
            for r in records if r.temperature_c is not None
        ]
        temperature_measurements.sort(key=lambda x: x["date"])
        
        # Анализ паттернов
        
        # Частые проблемы (топ симптомов/диагнозов)
        issues_counter = {}
        for r in records:
            if r.diagnosis:
                issues_counter[r.diagnosis] = issues_counter.get(r.diagnosis, 0) + 1
            elif r.symptoms:
                # Берём первые 50 символов как ключ
                key = r.symptoms[:50]
                issues_counter[key] = issues_counter.get(key, 0) + 1
        
        frequent_issues = sorted(issues_counter.items(), key=lambda x: x[1], reverse=True)[:3]
        frequent_issues = [issue for issue, count in frequent_issues if count > 1]
        
        # Тренд веса
        weight_trend = "insufficient_data"
        weight_change_kg = None
        weight_change_percent = None
        
        if len(weight_measurements) >= 2:
            first_weight = weight_measurements[0]["weight_kg"]
            last_weight = weight_measurements[-1]["weight_kg"]
            weight_change_kg = round(last_weight - first_weight, 2)
            weight_change_percent = round((weight_change_kg / first_weight * 100), 2) if first_weight > 0 else 0
            
            if abs(weight_change_percent) < 2:
                weight_trend = "stable"
            elif weight_change_kg > 0:
                weight_trend = "increasing"
            else:
                weight_trend = "decreasing"
        
        # Формируем результат
        result = {
            "pet_name": pet.name,
            "pet_species": pet.species,
            "period_days": period_days,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "total_records": len(records),
            "showing_unresolved_only": unresolved_only,
            "records": formatted_records,
            "statistics": {
                "by_type": by_type,
                "by_urgency": by_urgency,
                "unresolved_count": unresolved_count,
                "weight_measurements_count": len(weight_measurements),
                "temperature_measurements_count": len(temperature_measurements),
                "weight_measurements": weight_measurements,
                "temperature_measurements": temperature_measurements,
            },
            "patterns": {
                "frequent_issues": frequent_issues,
                "weight_trend": weight_trend,
                "weight_change_kg": weight_change_kg,
                "weight_change_percent": weight_change_percent,
            }
        }
        
        logger.info(f"Analyzed {len(records)} health records for {pet.name}")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to analyze health records: {e}")
        return json.dumps({
            "error": str(e),
            "pet_name": pet_name
        }, ensure_ascii=False)


@tool
async def calculate_daily_nutrition(
    pet_name: str,
    activity_level: Optional[str] = None,
) -> str:
    """Рассчитать суточную норму питания для питомца.
    
    Рассчитывает калории, белки, жиры, углеводы на основе веса, возраста,
    вида животного и уровня активности.
    
    Args:
        pet_name: Имя питомца
        activity_level: Уровень активности (низкий/средний/высокий, опционально)
    
    Returns:
        JSON с рекомендациями по питанию:
        {
          "pet_name": str,
          "weight_kg": float,
          "age_years": int,
          "species": str,
          "activity_level": str,
          "is_sterilized": bool,
          "daily_calories": {
            "min_kcal": float,
            "max_kcal": float,
            "recommended_kcal": float
          },
          "macronutrients": {
            "protein_g": float,
            "fat_g": float,
            "carbs_g": float
          },
          "feeding_schedule": {
            "meals_per_day": int,
            "portion_size_g": float,
            "note": str
          },
          "notes": [str],
          "calculated_at": ISO8601
        }
    """
    try:
        ctx = _get_context()
        pet_service = _get_pet_service()
        
        # Находим питомца
        user_pets = await pet_service.get_user_pets(ctx.user_id)
        pet = next((p for p in user_pets if p.name.lower() == pet_name.lower()), None)
        
        if not pet:
            return json.dumps({
                "error": f"Питомец '{pet_name}' не найден",
                "pet_name": pet_name
            }, ensure_ascii=False)
        
        # Проверяем вес
        if not pet.weight_kg:
            return json.dumps({
                "error": "Не указан вес питомца. Добавьте вес для расчёта питания.",
                "pet_name": pet.name
            }, ensure_ascii=False)
        
        # Определяем уровень активности
        activity = activity_level or pet.activity_level or "средний"
        activity = activity.lower()
        
        # Коэффициенты активности
        activity_multipliers = {
            "низкий": 1.2,
            "низкая": 1.2,
            "средний": 1.4,
            "средняя": 1.4,
            "высокий": 1.6,
            "высокая": 1.6,
            "очень высокий": 1.8,
            "очень высокая": 1.8,
        }
        
        multiplier = activity_multipliers.get(activity, 1.4)
        
        # Расчёт базового метаболизма (RER)
        # RER = 70 × (вес в кг)^0.75
        rer = 70 * (pet.weight_kg ** 0.75)
        
        # Суточная потребность в энергии (DER)
        der = rer * multiplier
        
        # Корректировка для возраста
        if pet.age_years:
            if pet.age_years < 1:
                # Котята/щенки
                der *= 1.5
            elif pet.age_years > 7:
                # Пожилые
                der *= 0.9
        
        # Корректировка для стерилизованных
        if pet.is_sterilized:
            der *= 0.9
        
        # Диапазон калорий (±10%)
        min_kcal = round(der * 0.9, 1)
        max_kcal = round(der * 1.1, 1)
        recommended_kcal = round(der, 1)
        
        # Макронутриенты
        if pet.species.lower() in ["кошка", "cat"]:
            # Кошки - облигатные хищники, больше белка
            protein_percent = 0.30
            fat_percent = 0.20
        else:
            # Собаки - всеядные
            protein_percent = 0.25
            fat_percent = 0.15
        
        protein_kcal = der * protein_percent
        fat_kcal = der * fat_percent
        carbs_kcal = der - protein_kcal - fat_kcal
        
        protein_g = round(protein_kcal / 4, 1)
        fat_g = round(fat_kcal / 9, 1)
        carbs_g = round(carbs_kcal / 4, 1)
        
        # График кормления
        if pet.age_years and pet.age_years < 1:
            meals_per_day = 3
        elif pet.weight_kg < 10:
            meals_per_day = 2
        else:
            meals_per_day = 2
        
        # Размер порции сухого корма (~350-400 ккал на 100г)
        portion_size_g = round(recommended_kcal / 3.8 / meals_per_day, 1)
        
        # Заметки
        notes = []
        
        if pet.age_years and pet.age_years < 1:
            notes.append("Котёнок/щенок - увеличенная норма для роста")
        
        if pet.age_years and pet.age_years > 7:
            notes.append("Пожилой питомец - сниженная калорийность")
        
        if pet.is_sterilized:
            notes.append("Стерилизован - снижен метаболизм на 10%")
        
        if pet.allergies:
            notes.append(f"Учитывайте аллергии: {pet.allergies}")
        
        if pet.chronic_conditions:
            notes.append(f"Учитывайте заболевания: {pet.chronic_conditions}")
        
        notes.append("Это общие рекомендации. Проконсультируйтесь с ветеринаром для точного рациона.")
        
        # Формируем результат
        result = {
            "pet_name": pet.name,
            "weight_kg": pet.weight_kg,
            "age_years": pet.age_years,
            "species": pet.species,
            "activity_level": activity,
            "is_sterilized": pet.is_sterilized,
            "daily_calories": {
                "min_kcal": min_kcal,
                "max_kcal": max_kcal,
                "recommended_kcal": recommended_kcal
            },
            "macronutrients": {
                "protein_g": protein_g,
                "fat_g": fat_g,
                "carbs_g": carbs_g
            },
            "feeding_schedule": {
                "meals_per_day": meals_per_day,
                "portion_size_g": portion_size_g,
                "note": "Примерный размер порции сухого корма (~370 ккал/100г)"
            },
            "notes": notes,
            "calculated_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Calculated nutrition for {pet.name}: {recommended_kcal} kcal/day")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to calculate nutrition: {e}")
        return json.dumps({
            "error": str(e),
            "pet_name": pet_name
        }, ensure_ascii=False)


@tool
async def analyze_food_ingredients(
    ingredient_list: str,
    pet_species: Optional[str] = None,
) -> str:
    """Проанализировать состав корма для питомца.
    
    Оценивает качество ингредиентов, выявляет потенциальные аллергены,
    проверяет соответствие потребностям вида животного.
    
    Args:
        ingredient_list: Список ингредиентов корма (через запятую или с новой строки)
        pet_species: Вид животного (кошка/собака, опционально)
    
    Returns:
        JSON с анализом состава:
        {
          "analyzed_at": ISO8601,
          "pet_species": str,
          "ingredients_count": int,
          "ingredients": [
            {
              "name": str,
              "position": int,
              "category": "protein|grain|vegetable|additive|unknown",
              "quality": "high|medium|low",
              "notes": str
            }
          ],
          "quality_assessment": {
            "overall_score": int (1-10),
            "protein_quality": "high|medium|low",
            "has_named_meat": bool,
            "grain_free": bool,
            "has_fillers": bool,
            "has_artificial_additives": bool
          },
          "warnings": [str],
          "recommendations": [str]
        }
    """
    try:
        ctx = _get_context()
        
        # Определяем вид животного
        species = pet_species
        if not species and ctx.current_pet_name:
            pet_service = _get_pet_service()
            user_pets = await pet_service.get_user_pets(ctx.user_id)
            pet = next((p for p in user_pets if p.name.lower() == ctx.current_pet_name.lower()), None)
            if pet:
                species = pet.species
        
        species = (species or "").lower()
        
        # Парсим список ингредиентов
        ingredients_raw = [
            ing.strip() 
            for ing in ingredient_list.replace('\n', ',').split(',')
            if ing.strip()
        ]
        
        if not ingredients_raw:
            return json.dumps({
                "error": "Не указаны ингредиенты для анализа",
            }, ensure_ascii=False)
        
        # Справочники для анализа
        
        # Качественные источники белка
        quality_proteins = [
            "курица", "индейка", "говядина", "ягнёнок", "рыба", "лосось", 
            "тунец", "утка", "кролик", "оленина", "chicken", "turkey", 
            "beef", "lamb", "fish", "salmon"
        ]
        
        # Низкокачественные источники белка
        low_quality_proteins = [
            "мясная мука", "мясные субпродукты", "животный жир", "костная мука",
            "meat meal", "meat by-products", "animal fat", "bone meal"
        ]
        
        # Зерновые
        grains = [
            "пшеница", "кукуруза", "рис", "ячмень", "овёс", "просо",
            "wheat", "corn", "rice", "barley", "oats", "millet"
        ]
        
        # Наполнители (филлеры)
        fillers = [
            "кукурузный глютен", "пшеничный глютен", "целлюлоза", "жом",
            "corn gluten", "wheat gluten", "cellulose", "beet pulp"
        ]
        
        # Искусственные добавки
        artificial_additives = [
            "BHA", "BHT", "этоксиквин", "красители", "ethoxyquin", "artificial colors"
        ]
        
        # Полезные добавки
        beneficial_additives = [
            "таурин", "витамин", "минерал", "omega", "жирные кислоты",
            "taurine", "vitamin", "mineral", "fatty acids", "пробиотик", "probiotic"
        ]
        
        # Анализируем каждый ингредиент
        analyzed_ingredients = []
        has_named_meat = False
        grain_free = True
        has_fillers = False
        has_artificial = False
        protein_quality = "low"
        
        for i, ing in enumerate(ingredients_raw, 1):
            ing_lower = ing.lower()
            
            # Определяем категорию и качество
            category = "unknown"
            quality = "medium"
            notes = []
            
            # Проверяем на качественный белок
            if any(protein in ing_lower for protein in quality_proteins):
                category = "protein"
                quality = "high"
                if i <= 3:  # Первые 3 ингредиента
                    has_named_meat = True
                    protein_quality = "high"
                notes.append("Качественный источник белка")
            
            # Низкокачественный белок
            elif any(protein in ing_lower for protein in low_quality_proteins):
                category = "protein"
                quality = "low"
                notes.append("Низкокачественный источник белка")
            
            # Зерновые
            elif any(grain in ing_lower for grain in grains):
                category = "grain"
                grain_free = False
                if i <= 3:
                    quality = "low"
                    notes.append("Зерновые в начале состава")
                else:
                    notes.append("Зерновой компонент")
            
            # Наполнители
            elif any(filler in ing_lower for filler in fillers):
                category = "filler"
                quality = "low"
                has_fillers = True
                notes.append("Наполнитель низкой питательной ценности")
            
            # Искусственные добавки
            elif any(additive in ing_lower for additive in artificial_additives):
                category = "additive"
                quality = "low"
                has_artificial = True
                notes.append("Искусственная добавка")
            
            # Полезные добавки
            elif any(additive in ing_lower for additive in beneficial_additives):
                category = "additive"
                quality = "high"
                notes.append("Полезная добавка")
            
            # Овощи/фрукты
            elif any(word in ing_lower for word in ["овощ", "фрукт", "ягод", "vegetable", "fruit", "berry"]):
                category = "vegetable"
                quality = "high"
                notes.append("Источник витаминов и клетчатки")
            
            analyzed_ingredients.append({
                "name": ing,
                "position": i,
                "category": category,
                "quality": quality,
                "notes": "; ".join(notes) if notes else None
            })
        
        # Общая оценка качества (1-10)
        score = 5  # Базовая оценка
        
        if has_named_meat:
            score += 2
        if protein_quality == "high":
            score += 1
        if grain_free:
            score += 1
        if not has_fillers:
            score += 1
        if not has_artificial:
            score += 1
        
        # Ограничиваем диапазон
        score = min(10, max(1, score))
        
        # Предупреждения
        warnings = []
        
        if not has_named_meat:
            warnings.append("Нет чётко указанного мясного ингредиента в начале состава")
        
        if has_fillers:
            warnings.append("Содержит наполнители низкой питательной ценности")
        
        if has_artificial:
            warnings.append("Содержит искусственные консерванты или красители")
        
        if not grain_free and species in ["кошка", "cat"]:
            warnings.append("Кошки - облигатные хищники, зерновые не являются их естественной пищей")
        
        # Рекомендации
        recommendations = []
        
        if score < 6:
            recommendations.append("Рассмотрите корма более высокого качества с указанием конкретного мяса в начале состава")
        
        if not grain_free:
            recommendations.append("Попробуйте беззерновые корма для лучшего пищеварения")
        
        if has_artificial:
            recommendations.append("Выбирайте корма с натуральными консервантами (токоферолы, розмарин)")
        
        if protein_quality == "low":
            recommendations.append("Ищите корма где первые 2-3 ингредиента - качественные источники белка")
        
        # Формируем результат
        result = {
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "pet_species": species if species else "не указан",
            "ingredients_count": len(analyzed_ingredients),
            "ingredients": analyzed_ingredients,
            "quality_assessment": {
                "overall_score": score,
                "score_description": _get_score_description(score),
                "protein_quality": protein_quality,
                "has_named_meat": has_named_meat,
                "grain_free": grain_free,
                "has_fillers": has_fillers,
                "has_artificial_additives": has_artificial,
            },
            "warnings": warnings,
            "recommendations": recommendations
        }
        
        logger.info(f"Analyzed food ingredients: score={score}/10, ingredients={len(analyzed_ingredients)}")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to analyze food ingredients: {e}")
        return json.dumps({
            "error": str(e),
        }, ensure_ascii=False)


def _get_score_description(score: int) -> str:
    """Получить текстовое описание оценки"""
    if score >= 9:
        return "Отличный корм"
    elif score >= 7:
        return "Хороший корм"
    elif score >= 5:
        return "Средний корм"
    elif score >= 3:
        return "Низкое качество"
    else:
        return "Очень низкое качество"


@tool
async def check_vaccination_schedule(pet_name: str) -> str:
    """Проверить график прививок питомца.
    
    Анализирует сделанные прививки и показывает что нужно сделать.
    
    Args:
        pet_name: Имя питомца
    
    Returns:
        JSON с информацией о прививках:
        {
          "pet_name": str,
          "species": str,
          "age_years": int,
          "vaccinations_done": [
            {"date": str, "name": str, "next_due": str, "clinic": str}
          ],
          "vaccinations_needed": [
            {"name": str, "recommended_age": str, "priority": str}
          ],
          "overdue": [
            {"name": str, "last_date": str, "overdue_days": int, "priority": str}
          ],
          "checked_at": ISO8601
        }
    """
    try:
        ctx = _get_context()
        pet_service = _get_pet_service()
        health_service = _get_health_service()
        
        # Находим питомца
        user_pets = await pet_service.get_user_pets(ctx.user_id)
        pet = next((p for p in user_pets if p.name.lower() == pet_name.lower()), None)
        
        if not pet:
            return json.dumps({
                "error": f"Питомец '{pet_name}' не найден",
                "pet_name": pet_name
            }, ensure_ascii=False)
        
        # Получаем записи о прививках
        all_records = await health_service.get_pet_health_records(
            pet_id=pet.id,
            user_id=ctx.user_id
        )
        
        vaccinations = [r for r in all_records if r.record_type.value == "vaccination"]
        vaccinations.sort(key=lambda x: x.record_date, reverse=True)
        
        # Сделанные прививки
        vaccinations_done = []
        for vacc in vaccinations:
            next_due = vacc.next_visit_date.isoformat() if vacc.next_visit_date else (vacc.record_date + timedelta(days=365)).isoformat()
            
            vaccinations_done.append({
                "date": vacc.record_date.isoformat(),
                "name": vacc.title,
                "description": vacc.description,
                "next_due": next_due,
                "clinic": vacc.vet_clinic
            })
        
        # Рекомендуемые прививки
        if pet.species.lower() in ["кошка", "cat"]:
            recommended = [
                {"name": "Бешенство", "interval_days": 365, "priority": "critical"},
                {"name": "Панлейкопения", "interval_days": 365, "priority": "high"},
                {"name": "Калицивироз", "interval_days": 365, "priority": "high"},
                {"name": "Ринотрахеит", "interval_days": 365, "priority": "high"},
            ]
        else:
            recommended = [
                {"name": "Бешенство", "interval_days": 365, "priority": "critical"},
                {"name": "Чума", "interval_days": 365, "priority": "critical"},
                {"name": "Парвовирус", "interval_days": 365, "priority": "high"},
                {"name": "Аденовироз", "interval_days": 365, "priority": "high"},
                {"name": "Лептоспироз", "interval_days": 365, "priority": "medium"},
            ]
        
        # Просроченные
        overdue = []
        today = date.today()
        
        for rec in recommended:
            matching = [v for v in vaccinations if rec["name"].lower() in v.title.lower()]
            
            if matching:
                last_vacc = matching[0]
                next_due_date = last_vacc.record_date + timedelta(days=rec["interval_days"])
                
                if next_due_date < today:
                    overdue.append({
                        "name": rec["name"],
                        "last_date": last_vacc.record_date.isoformat(),
                        "overdue_days": (today - next_due_date).days,
                        "priority": rec["priority"]
                    })
        
        # Нужные прививки
        vaccinations_needed = []
        for rec in recommended:
            has_vacc = any(rec["name"].lower() in v.title.lower() for v in vaccinations)
            
            if not has_vacc:
                age_rec = "с 2 месяцев" if pet.age_years and pet.age_years < 1 else "как можно скорее"
                vaccinations_needed.append({
                    "name": rec["name"],
                    "recommended_age": age_rec,
                    "priority": rec["priority"]
                })
        
        result = {
            "pet_name": pet.name,
            "species": pet.species,
            "age_years": pet.age_years,
            "vaccinations_done": vaccinations_done,
            "vaccinations_needed": vaccinations_needed,
            "overdue": overdue,
            "total_vaccinations": len(vaccinations),
            "checked_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Checked vaccinations for {pet.name}: done={len(vaccinations_done)}, overdue={len(overdue)}")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to check vaccination schedule: {e}")
        return json.dumps({
            "error": str(e),
            "pet_name": pet_name
        }, ensure_ascii=False)


# ============================================================================
# HEALTH & NUTRITION AGENT
# ============================================================================

class HealthNutritionAgent:
    """Агент для анализа здоровья и питания питомцев"""
    
    def __init__(
        self,
        pet_service: PetService,
        health_record_service: HealthRecordService,
        llm=None
    ):
        self.pet_service = pet_service
        self.health_record_service = health_record_service
        self.llm = llm or GigaChatClient().llm
        
        # Список инструментов
        self.tools = [
            analyze_health_records,
            calculate_daily_nutrition,
            analyze_food_ingredients,
            check_vaccination_schedule,
        ]
        
        logger.info("HealthNutritionAgent initialized with 4 tools")
    
    async def process(
        self,
        user_id: int,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Обработать запрос пользователя"""
        context = context or {}
        ctx_token = None
        pet_token = None
        health_token = None
        
        try:
            tool_context = HealthNutritionContext(
                user_id=user_id,
                current_pet_id=context.get("current_pet_id"),
                current_pet_name=context.get("current_pet_name", ""),
            )
            
            ctx_token = _health_nutrition_context.set(tool_context)
            pet_token = _pet_service.set(self.pet_service)
            health_token = _health_service.set(self.health_record_service)
            
            user_pets = await self.pet_service.get_user_pets(user_id)
            pets_info = ""
            if user_pets:
                pets_list = [f"{p.name} ({p.species})" for p in user_pets]
                pets_info = f"\n🐾 Питомцы: {', '.join(pets_list)}"
            
            system_prompt = f"""Ты - эксперт по здоровью и питанию ДОМАШНИХ ЖИВОТНЫХ (кошки, собаки, и другие питомцы).

Пользователь ID: {user_id}{pets_info}

**ВАЖНО:** Ты работаешь ТОЛЬКО с домашними животными. Если вопрос НЕ о питомце (например, о растениях, о людях, о садоводстве), сообщи:
"Извините, я специализируюсь только на здоровье и питании домашних животных. Для вопросов о растениях или других темах обратитесь к другим специалистам."

**Доступные инструменты (4) - для ПИТОМЦЕВ:**

1. **analyze_health_records** - Анализ медицинских записей ПИТОМЦА
   Используй: "История болезней кота", "Анализ здоровья собаки", "Динамика веса питомца"

2. **calculate_daily_nutrition** - Расчёт суточной нормы питания для ЖИВОТНОГО
   Используй: "Сколько кормить кошку", "Норма калорий для собаки", "Рацион хомяка"

3. **analyze_food_ingredients** - Анализ состава КОРМА для питомца
   Используй: "Проверь состав корма для собаки", "Хороший ли корм для кошки"

4. **check_vaccination_schedule** - График прививок ПИТОМЦА
   Используй: "Какие прививки нужны коту", "Когда прививка собаке", "График вакцинации"

Все инструменты возвращают JSON для оркестратора. Анализируй данные профессионально!"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=settings.DEBUG,
                handle_parsing_errors=True,
                max_iterations=5,
            )
            
            result = await agent_executor.ainvoke({"input": user_message})
            return result.get("output", '{"error": "No output"}')
            
        except Exception as e:
            logger.exception(f"HealthNutritionAgent error for user {user_id}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        finally:
            if ctx_token:
                _health_nutrition_context.reset(ctx_token)
            if pet_token:
                _pet_service.reset(pet_token)
            if health_token:
                _health_service.reset(health_token)