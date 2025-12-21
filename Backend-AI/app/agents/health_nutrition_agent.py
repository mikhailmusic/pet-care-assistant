from __future__ import annotations

from typing import Optional, Annotated, List
from datetime import datetime, date, timedelta, timezone
from loguru import logger
import json

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from app.services.pet_service import PetService
from app.services.health_record_service import HealthRecordService


def _get_score_description(score: int) -> str:
    """Получить текстовое описание оценки качества корма"""
    if score >= 9:
        return "Отличный корм премиум-класса"
    elif score >= 7:
        return "Хороший корм"
    elif score >= 5:
        return "Средний корм"
    elif score >= 3:
        return "Корм низкого качества"
    else:
        return "Корм очень низкого качества"


class HealthNutritionTools:
    """Инкапсулирует tools для анализа здоровья и питания питомцев"""
    
    def __init__(
        self,
        pet_service: PetService,
        health_service: HealthRecordService,
    ):
        self.pet_service = pet_service
        self.health_service = health_service
    
    @tool
    async def analyze_health_records(
        self,
        state: Annotated[dict, InjectedState],
        pet_name: str,
        period_days: int = 90,
        unresolved_only: bool = False,
        max_records: int = 50,
    ) -> str:
        """Анализировать медицинские записи питомца за период.
        
        Собирает и анализирует все медицинские записи: симптомы, диагнозы, лечение,
        прививки, анализы. Выявляет паттерны и тренды.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            pet_name: Имя питомца
            period_days: Период анализа в днях (по умолчанию 90)
            unresolved_only: Показать только нерешённые проблемы
            max_records: Максимальное количество записей для анализа
        
        Returns:
            JSON с полным анализом медицинских записей
        """
        try:
            user_id = state["user_id"]
            
            # Находим питомца
            user_pets = await self.pet_service.get_user_pets(user_id)
            pet = next((p for p in user_pets if p.name.lower() == pet_name.lower()), None)
            
            if not pet:
                available = ", ".join([p.name for p in user_pets]) if user_pets else "нет"
                return json.dumps({
                    "error": f"Питомец '{pet_name}' не найден",
                    "available_pets": available
                }, ensure_ascii=False)
            
            # Получаем медицинские записи за период
            cutoff_date = date.today() - timedelta(days=period_days)
            all_records = await self.health_service.get_pet_health_records(
                pet_id=pet.id,
                user_id=user_id
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
            
            # === СТАТИСТИКА ===
            
            # По типам записей
            by_type = {}
            for r in records:
                record_type = r.record_type.value
                by_type[record_type] = by_type.get(record_type, 0) + 1
            
            # По срочности
            by_urgency = {
                "critical": sum(1 for r in records if r.urgency.value == "critical"),
                "high": sum(1 for r in records if r.urgency.value == "high"),
                "medium": sum(1 for r in records if r.urgency.value == "medium"),
                "low": sum(1 for r in records if r.urgency.value == "low"),
            }
            
            # Нерешённые проблемы
            unresolved_count = sum(1 for r in records if not r.is_resolved)
            unresolved_critical = sum(1 for r in records if not r.is_resolved and r.urgency.value in ["critical", "high"])
            
            # === ИЗМЕРЕНИЯ ===
            
            # Вес
            weight_measurements = [
                {"date": r.record_date.isoformat(), "weight_kg": r.weight_kg}
                for r in records if r.weight_kg is not None
            ]
            weight_measurements.sort(key=lambda x: x["date"])
            
            # Температура
            temperature_measurements = [
                {"date": r.record_date.isoformat(), "temperature_c": r.temperature_c}
                for r in records if r.temperature_c is not None
            ]
            temperature_measurements.sort(key=lambda x: x["date"])
            
            # === АНАЛИЗ ПАТТЕРНОВ ===
            
            # Частые проблемы
            issues_counter = {}
            for r in records:
                if r.diagnosis:
                    issues_counter[r.diagnosis] = issues_counter.get(r.diagnosis, 0) + 1
                elif r.symptoms:
                    # Берём первые 50 символов как ключ
                    key = r.symptoms[:50]
                    issues_counter[key] = issues_counter.get(key, 0) + 1
            
            frequent_issues = sorted(issues_counter.items(), key=lambda x: x[1], reverse=True)[:5]
            frequent_issues = [{"issue": issue, "count": count} for issue, count in frequent_issues if count > 1]
            
            # Тренд веса
            weight_trend = "insufficient_data"
            weight_change_kg = None
            weight_change_percent = None
            weight_trend_description = None
            
            if len(weight_measurements) >= 2:
                first_weight = weight_measurements[0]["weight_kg"]
                last_weight = weight_measurements[-1]["weight_kg"]
                weight_change_kg = round(last_weight - first_weight, 2)
                weight_change_percent = round((weight_change_kg / first_weight * 100), 2) if first_weight > 0 else 0
                
                if abs(weight_change_percent) < 2:
                    weight_trend = "stable"
                    weight_trend_description = "Вес стабилен"
                elif weight_change_kg > 0:
                    weight_trend = "increasing"
                    if weight_change_percent > 10:
                        weight_trend_description = f"Значительный набор веса (+{weight_change_percent}%)"
                    else:
                        weight_trend_description = f"Набор веса (+{weight_change_percent}%)"
                else:
                    weight_trend = "decreasing"
                    if weight_change_percent < -10:
                        weight_trend_description = f"Значительная потеря веса ({weight_change_percent}%)"
                    else:
                        weight_trend_description = f"Потеря веса ({weight_change_percent}%)"
            
            # Тренд температуры
            temperature_trend = "normal"
            avg_temperature = None
            abnormal_temps = []
            
            if temperature_measurements:
                temps = [t["temperature_c"] for t in temperature_measurements]
                avg_temperature = round(sum(temps) / len(temps), 1)
                
                # Нормальная температура для животных: 37.5-39.2°C
                abnormal_temps = [t for t in temperature_measurements if t["temperature_c"] < 37.0 or t["temperature_c"] > 39.5]
                
                if abnormal_temps:
                    temperature_trend = "abnormal"
            
            # Анализ затрат
            total_cost = sum(r.cost for r in records if r.cost)
            avg_cost_per_visit = round(total_cost / len(records), 2) if records else 0
            
            # === РЕКОМЕНДАЦИИ ===
            
            recommendations = []
            
            if unresolved_critical > 0:
                recommendations.append(f"⚠️ У питомца {unresolved_critical} нерешённых проблем высокой важности - требуется внимание!")
            
            if weight_trend == "increasing" and weight_change_percent and weight_change_percent > 10:
                recommendations.append("⚠️ Значительный набор веса - рекомендуется скорректировать рацион и увеличить активность")
            
            if weight_trend == "decreasing" and weight_change_percent and weight_change_percent < -10:
                recommendations.append("⚠️ Значительная потеря веса - необходима консультация ветеринара")
            
            if abnormal_temps:
                recommendations.append(f"⚠️ Обнаружено {len(abnormal_temps)} случаев аномальной температуры - следите за состоянием питомца")
            
            if frequent_issues:
                top_issue = frequent_issues[0]["issue"]
                recommendations.append(f"Частая проблема: '{top_issue}' - рекомендуется профилактика")
            
            # === ФОРМИРУЕМ РЕЗУЛЬТАТ ===
            
            result = {
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "pet": {
                    "name": pet.name,
                    "species": pet.species,
                    "breed": pet.breed,
                    "age_years": pet.age_years,
                    "current_weight_kg": pet.weight_kg,
                },
                "period": {
                    "days": period_days,
                    "from_date": cutoff_date.isoformat(),
                    "to_date": date.today().isoformat(),
                },
                "summary": {
                    "total_records": len(records),
                    "showing_unresolved_only": unresolved_only,
                    "unresolved_count": unresolved_count,
                    "unresolved_critical": unresolved_critical,
                    "total_cost": total_cost,
                    "avg_cost_per_visit": avg_cost_per_visit,
                },
                "records": formatted_records,
                "statistics": {
                    "by_type": by_type,
                    "by_urgency": by_urgency,
                    "measurements": {
                        "weight_count": len(weight_measurements),
                        "temperature_count": len(temperature_measurements),
                        "weight_data": weight_measurements,
                        "temperature_data": temperature_measurements,
                    }
                },
                "patterns": {
                    "frequent_issues": frequent_issues,
                    "weight_trend": {
                        "trend": weight_trend,
                        "description": weight_trend_description,
                        "change_kg": weight_change_kg,
                        "change_percent": weight_change_percent,
                    },
                    "temperature_trend": {
                        "status": temperature_trend,
                        "avg_temperature": avg_temperature,
                        "abnormal_count": len(abnormal_temps),
                        "abnormal_cases": abnormal_temps,
                    }
                },
                "recommendations": recommendations,
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
        self,
        state: Annotated[dict, InjectedState],
        pet_name: str,
        activity_level: Optional[str] = None,
    ) -> str:
        """Рассчитать суточную норму питания для питомца.
        
        Рассчитывает калории, белки, жиры, углеводы на основе веса, возраста,
        вида животного и уровня активности.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            pet_name: Имя питомца
            activity_level: Уровень активности (низкий/средний/высокий/очень_высокий)
        
        Returns:
            JSON с детальными рекомендациями по питанию
        """
        try:
            user_id = state["user_id"]
            
            # Находим питомца
            user_pets = await self.pet_service.get_user_pets(user_id)
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
            
            # Коэффициенты активности (улучшенные)
            activity_multipliers = {
                "низкий": 1.2,
                "низкая": 1.2,
                "средний": 1.4,
                "средняя": 1.4,
                "высокий": 1.6,
                "высокая": 1.6,
                "очень высокий": 1.8,
                "очень высокая": 1.8,
                "очень_высокий": 1.8,
                "очень_высокая": 1.8,
            }
            
            multiplier = activity_multipliers.get(activity, 1.4)
            
            # === РАСЧЁТ БАЗОВОГО МЕТАБОЛИЗМА (RER) ===
            # RER = 70 × (вес в кг)^0.75
            rer = 70 * (pet.weight_kg ** 0.75)
            
            # === РАСЧЁТ СУТОЧНОЙ ПОТРЕБНОСТИ (DER) ===
            der = rer * multiplier
            
            # Корректировка для возраста
            age_adjustment = 1.0
            age_note = None
            
            if pet.age_years:
                if pet.age_years < 1:
                    # Котята/щенки - повышенная энергия для роста
                    if pet.age_years < 0.5:  # До 6 месяцев
                        age_adjustment = 2.0
                        age_note = "Активный рост (до 6 мес) - удвоенная энергия"
                    else:
                        age_adjustment = 1.5
                        age_note = "Рост (6-12 мес) - повышенная энергия"
                elif pet.age_years > 7:
                    # Пожилые животные - снижен метаболизм
                    if pet.age_years > 10:
                        age_adjustment = 0.8
                        age_note = "Пожилой возраст (10+ лет) - снижен метаболизм"
                    else:
                        age_adjustment = 0.9
                        age_note = "Зрелый возраст (7-10 лет) - немного снижен метаболизм"
            
            der *= age_adjustment
            
            # Корректировка для стерилизованных
            sterilization_adjustment = 1.0
            sterilization_note = None
            
            if pet.is_sterilized:
                sterilization_adjustment = 0.85
                sterilization_note = "Стерилизован - снижен метаболизм на 15%"
                der *= sterilization_adjustment
            
            # Диапазон калорий (±10%)
            min_kcal = round(der * 0.9, 1)
            max_kcal = round(der * 1.1, 1)
            recommended_kcal = round(der, 1)
            
            # === МАКРОНУТРИЕНТЫ ===
            
            # Определяем соотношение макронутриентов по виду
            if pet.species.lower() in ["кошка", "cat", "кот"]:
                # Кошки - облигатные хищники
                protein_percent = 0.35  # 35% белка
                fat_percent = 0.20      # 20% жира
                carbs_percent = 0.05    # 5% углеводов (минимум)
            elif pet.species.lower() in ["собака", "dog", "пёс"]:
                # Собаки - всеядные
                protein_percent = 0.28  # 28% белка
                fat_percent = 0.17      # 17% жира
                carbs_percent = 0.15    # 15% углеводов
            else:
                # Другие животные - средние значения
                protein_percent = 0.25
                fat_percent = 0.15
                carbs_percent = 0.10
            
            # Расчёт граммов (1г белка = 4 ккал, 1г жира = 9 ккал, 1г углеводов = 4 ккал)
            protein_kcal = der * protein_percent
            fat_kcal = der * fat_percent
            carbs_kcal = der * carbs_percent
            
            protein_g = round(protein_kcal / 4, 1)
            fat_g = round(fat_kcal / 9, 1)
            carbs_g = round(carbs_kcal / 4, 1)
            
            # === ГРАФИК КОРМЛЕНИЯ ===
            
            if pet.age_years and pet.age_years < 0.5:
                meals_per_day = 4
                feeding_note = "Частое кормление малыми порциями для котят/щенков"
            elif pet.age_years and pet.age_years < 1:
                meals_per_day = 3
                feeding_note = "3 раза в день для растущего организма"
            elif pet.weight_kg < 5:
                meals_per_day = 2
                feeding_note = "2 раза в день для мелких пород"
            else:
                meals_per_day = 2
                feeding_note = "2 раза в день - стандартный режим"
            
            # Размер порции сухого корма
            # Средняя калорийность сухого корма: ~370 ккал/100г
            dry_food_kcal_per_100g = 370
            daily_dry_food_g = round(recommended_kcal / dry_food_kcal_per_100g * 100, 1)
            portion_size_g = round(daily_dry_food_g / meals_per_day, 1)
            
            # Размер порции влажного корма
            # Средняя калорийность влажного корма: ~80 ккал/100г
            wet_food_kcal_per_100g = 80
            daily_wet_food_g = round(recommended_kcal / wet_food_kcal_per_100g * 100, 1)
            wet_portion_size_g = round(daily_wet_food_g / meals_per_day, 1)
            
            # === СПЕЦИАЛЬНЫЕ РЕКОМЕНДАЦИИ ===
            
            recommendations = []
            warnings = []
            
            # На основе возраста
            if age_note:
                recommendations.append(age_note)
            
            # На основе стерилизации
            if sterilization_note:
                recommendations.append(sterilization_note)
            
            # На основе здоровья
            if pet.allergies:
                warnings.append(f"⚠️ Аллергии: {pet.allergies} - выбирайте гипоаллергенный корм")
            
            if pet.chronic_conditions:
                warnings.append(f"⚠️ Заболевания: {pet.chronic_conditions} - требуется специальная диета")
            
            if pet.medications:
                warnings.append(f"⚠️ Принимает лекарства: {pet.medications} - учитывайте совместимость с кормом")
            
            # Проверка веса
            if pet.species.lower() in ["кошка", "cat", "кот"]:
                # Средний вес кошки: 4-5 кг
                if pet.weight_kg > 6:
                    warnings.append("⚠️ Избыточный вес - рекомендуется диета и увеличение активности")
                elif pet.weight_kg < 3 and pet.age_years and pet.age_years > 1:
                    warnings.append("⚠️ Недостаточный вес - возможно требуется усиленное питание")
            
            # Общие рекомендации
            recommendations.append("Всегда обеспечивайте свежую питьевую воду")
            recommendations.append("Кормите в одно и то же время каждый день")
            recommendations.append("Не перекармливайте - ожирение опасно для здоровья")
            recommendations.append("Проконсультируйтесь с ветеринаром для точного подбора рациона")
            
            # === ФОРМИРУЕМ РЕЗУЛЬТАТ ===
            
            result = {
                "calculated_at": datetime.now(timezone.utc).isoformat(),
                "pet": {
                    "name": pet.name,
                    "species": pet.species,
                    "breed": pet.breed,
                    "weight_kg": pet.weight_kg,
                    "age_years": pet.age_years,
                    "age_months": pet.age_months,
                    "is_sterilized": pet.is_sterilized,
                    "activity_level": activity,
                },
                "calculations": {
                    "rer_kcal": round(rer, 1),
                    "activity_multiplier": multiplier,
                    "age_adjustment": age_adjustment,
                    "sterilization_adjustment": sterilization_adjustment,
                },
                "daily_calories": {
                    "min_kcal": min_kcal,
                    "recommended_kcal": recommended_kcal,
                    "max_kcal": max_kcal,
                },
                "macronutrients": {
                    "protein": {
                        "grams": protein_g,
                        "percent": round(protein_percent * 100, 1),
                        "note": "Основной строительный материал организма"
                    },
                    "fat": {
                        "grams": fat_g,
                        "percent": round(fat_percent * 100, 1),
                        "note": "Энергия и усвоение витаминов"
                    },
                    "carbs": {
                        "grams": carbs_g,
                        "percent": round(carbs_percent * 100, 1),
                        "note": "Источник энергии (минимум для хищников)"
                    }
                },
                "feeding_schedule": {
                    "meals_per_day": meals_per_day,
                    "note": feeding_note,
                    "dry_food": {
                        "total_daily_g": daily_dry_food_g,
                        "portion_per_meal_g": portion_size_g,
                        "note": f"Сухой корм (~{dry_food_kcal_per_100g} ккал/100г)"
                    },
                    "wet_food": {
                        "total_daily_g": daily_wet_food_g,
                        "portion_per_meal_g": wet_portion_size_g,
                        "note": f"Влажный корм (~{wet_food_kcal_per_100g} ккал/100г)"
                    }
                },
                "recommendations": recommendations,
                "warnings": warnings,
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
        self,
        state: Annotated[dict, InjectedState],
        ingredient_list: str,
        pet_species: Optional[str] = None,
    ) -> str:
        """Проанализировать состав корма для питомца.
        
        Оценивает качество ингредиентов, выявляет потенциальные аллергены,
        проверяет соответствие потребностям вида животного.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            ingredient_list: Список ингредиентов корма (через запятую или с новой строки)
            pet_species: Вид животного (кошка/собака, опционально)
        
        Returns:
            JSON с детальным анализом состава корма
        """
        try:
            user_id = state["user_id"]
            
            # Определяем вид животного
            species = (pet_species or "").lower()
            
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
            
            # === СПРАВОЧНИКИ ДЛЯ АНАЛИЗА ===
            
            # Качественные источники белка
            quality_proteins = {
                "курица": "high", "индейка": "high", "говядина": "high", "ягнёнок": "high",
                "рыба": "high", "лосось": "high", "тунец": "high", "утка": "high",
                "кролик": "high", "оленина": "high", "перепел": "high",
                "chicken": "high", "turkey": "high", "beef": "high", "lamb": "high",
                "fish": "high", "salmon": "high", "duck": "high", "rabbit": "high"
            }
            
            # Низкокачественные источники белка
            low_quality_proteins = {
                "мясная мука": "low", "мясные субпродукты": "low", "животный жир": "low",
                "костная мука": "low", "кровяная мука": "low", "мясокостная мука": "low",
                "meat meal": "low", "meat by-products": "low", "animal fat": "low",
                "bone meal": "low", "blood meal": "low"
            }
            
            # Зерновые
            grains = [
                "пшеница", "кукуруза", "рис", "ячмень", "овёс", "просо", "сорго",
                "wheat", "corn", "rice", "barley", "oats", "millet", "sorghum"
            ]
            
            # Наполнители (филлеры)
            fillers = [
                "кукурузный глютен", "пшеничный глютен", "целлюлоза", "жом",
                "соевая мука", "corn gluten", "wheat gluten", "cellulose",
                "beet pulp", "soy flour"
            ]
            
            # Искусственные добавки (вредные)
            artificial_additives = [
                "BHA", "BHT", "этоксиквин", "пропилгаллат", "искусственные красители",
                "ethoxyquin", "propyl gallate", "artificial colors", "artificial flavors"
            ]
            
            # Полезные добавки
            beneficial_additives = [
                "таурин", "l-карнитин", "глюкозамин", "хондроитин", "омега-3", "омега-6",
                "витамин", "минерал", "пробиотик", "пребиотик", "антиоксидант",
                "taurine", "l-carnitine", "glucosamine", "chondroitin", "omega",
                "vitamin", "mineral", "probiotic", "prebiotic", "antioxidant"
            ]
            
            # Потенциальные аллергены
            common_allergens = [
                "пшеница", "кукуруза", "соя", "молочные продукты", "яйца",
                "wheat", "corn", "soy", "dairy", "eggs"
            ]
            
            # === АНАЛИЗ КАЖДОГО ИНГРЕДИЕНТА ===
            
            analyzed_ingredients = []
            has_named_meat = False
            grain_free = True
            has_fillers = False
            has_artificial = False
            protein_quality = "low"
            allergens_found = []
            beneficial_count = 0
            
            for i, ing in enumerate(ingredients_raw, 1):
                ing_lower = ing.lower()
                
                category = "unknown"
                quality = "medium"
                notes = []
                is_allergen = False
                
                # Проверяем на качественный белок
                for protein, qual in quality_proteins.items():
                    if protein in ing_lower:
                        category = "protein"
                        quality = qual
                        if i <= 3:
                            has_named_meat = True
                            protein_quality = "high"
                        notes.append("Качественный источник белка")
                        break
                
                # Низкокачественный белок
                if category == "unknown":
                    for protein, qual in low_quality_proteins.items():
                        if protein in ing_lower:
                            category = "protein"
                            quality = qual
                            notes.append("Низкокачественный источник белка (субпродукты)")
                            break
                
                # Зерновые
                if category == "unknown":
                    if any(grain in ing_lower for grain in grains):
                        category = "grain"
                        grain_free = False
                        if i <= 3:
                            quality = "low"
                            notes.append("Зерновые в начале состава - много углеводов")
                        else:
                            notes.append("Зерновой компонент")
                
                # Наполнители
                if any(filler in ing_lower for filler in fillers):
                    category = "filler"
                    quality = "low"
                    has_fillers = True
                    notes.append("Наполнитель низкой питательной ценности")
                
                # Искусственные добавки
                if any(additive in ing_lower for additive in artificial_additives):
                    category = "additive"
                    quality = "low"
                    has_artificial = True
                    notes.append("⚠️ Искусственная добавка (потенциально вредна)")
                
                # Полезные добавки
                if any(additive in ing_lower for additive in beneficial_additives):
                    if category == "unknown":
                        category = "additive"
                    quality = "high"
                    beneficial_count += 1
                    notes.append("✅ Полезная добавка")
                
                # Овощи/фрукты
                if category == "unknown":
                    if any(word in ing_lower for word in ["овощ", "фрукт", "ягод", "морков", "тыкв", "яблок", "vegetable", "fruit", "berry", "carrot", "pumpkin", "apple"]):
                        category = "vegetable"
                        quality = "high"
                        notes.append("Источник витаминов и клетчатки")
                
                # Проверка на аллергены
                if any(allergen in ing_lower for allergen in common_allergens):
                    is_allergen = True
                    allergens_found.append(ing)
                
                analyzed_ingredients.append({
                    "name": ing,
                    "position": i,
                    "category": category,
                    "quality": quality,
                    "is_allergen": is_allergen,
                    "notes": "; ".join(notes) if notes else None
                })
            
            # === ОБЩАЯ ОЦЕНКА КАЧЕСТВА (1-10) ===
            
            score = 5  # Базовая оценка
            
            # Белок в начале состава
            if has_named_meat:
                score += 2
            if protein_quality == "high":
                score += 1
            
            # Беззерновой
            if grain_free:
                score += 1
            
            # Нет наполнителей
            if not has_fillers:
                score += 1
            
            # Нет искусственных добавок
            if not has_artificial:
                score += 1
            
            # Много полезных добавок
            if beneficial_count >= 3:
                score += 1
            
            # Специфично для вида
            if species in ["кошка", "cat", "кот"]:
                # Кошки нуждаются в таурине
                has_taurine = any("таурин" in ing.lower() or "taurine" in ing.lower() for ing in ingredients_raw)
                if has_taurine:
                    score += 1
                else:
                    score -= 1
            
            # Ограничиваем диапазон
            score = min(10, max(1, score))
            
            # === ПРЕДУПРЕЖДЕНИЯ ===
            
            warnings = []
            
            if not has_named_meat:
                warnings.append("⚠️ Нет чётко указанного мясного ингредиента в начале состава")
            
            if has_fillers:
                warnings.append("⚠️ Содержит наполнители низкой питательной ценности")
            
            if has_artificial:
                warnings.append("⚠️ Содержит искусственные консерванты или красители")
            
            if not grain_free and species in ["кошка", "cat", "кот"]:
                warnings.append("⚠️ Кошки - облигатные хищники, зерновые не являются их естественной пищей")
            
            if allergens_found:
                warnings.append(f"⚠️ Обнаружены потенциальные аллергены: {', '.join(allergens_found)}")
            
            if species in ["кошка", "cat", "кот"]:
                has_taurine = any("таурин" in ing.lower() or "taurine" in ing.lower() for ing in ingredients_raw)
                if not has_taurine:
                    warnings.append("⚠️ Не обнаружен таурин - критически важен для кошек!")
            
            # === РЕКОМЕНДАЦИИ ===
            
            recommendations = []
            
            if score < 6:
                recommendations.append("Рассмотрите корма более высокого качества с указанием конкретного мяса в начале состава")
            
            if not grain_free:
                recommendations.append("Попробуйте беззерновые корма для лучшего пищеварения")
            
            if has_artificial:
                recommendations.append("Выбирайте корма с натуральными консервантами (токоферолы, розмарин)")
            
            if protein_quality == "low":
                recommendations.append("Ищите корма где первые 2-3 ингредиента - качественные источники белка (курица, говядина, рыба)")
            
            if allergens_found:
                recommendations.append("Если у питомца есть аллергии, выбирайте гипоаллергенные корма")
            
            recommendations.append("Всегда читайте состав - качество корма определяется первыми 5-7 ингредиентами")
            
            # === ФОРМИРУЕМ РЕЗУЛЬТАТ ===
            
            result = {
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
                "pet_species": species if species else "не указан",
                "ingredients_count": len(analyzed_ingredients),
                "ingredients": analyzed_ingredients,
                "quality_assessment": {
                    "overall_score": score,
                    "max_score": 10,
                    "score_description": _get_score_description(score),
                    "protein_quality": protein_quality,
                    "has_named_meat": has_named_meat,
                    "grain_free": grain_free,
                    "has_fillers": has_fillers,
                    "has_artificial_additives": has_artificial,
                    "beneficial_additives_count": beneficial_count,
                },
                "allergens": {
                    "found": len(allergens_found) > 0,
                    "list": allergens_found,
                },
                "warnings": warnings,
                "recommendations": recommendations,
            }
            
            logger.info(f"Analyzed food ingredients: score={score}/10, ingredients={len(analyzed_ingredients)}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to analyze food ingredients: {e}")
            return json.dumps({
                "error": str(e),
            }, ensure_ascii=False)
    
    @tool
    async def check_vaccination_schedule(
        self,
        state: Annotated[dict, InjectedState],
        pet_name: str
    ) -> str:
        """Проверить график прививок питомца.
        
        Анализирует сделанные прививки и показывает что нужно сделать.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            pet_name: Имя питомца
        
        Returns:
            JSON с полной информацией о прививках
        """
        try:
            user_id = state["user_id"]
            
            # Находим питомца
            user_pets = await self.pet_service.get_user_pets(user_id)
            pet = next((p for p in user_pets if p.name.lower() == pet_name.lower()), None)
            
            if not pet:
                return json.dumps({
                    "error": f"Питомец '{pet_name}' не найден",
                    "pet_name": pet_name
                }, ensure_ascii=False)
            
            # Получаем записи о прививках
            all_records = await self.health_service.get_pet_health_records(
                pet_id=pet.id,
                user_id=user_id
            )
            
            vaccinations = [r for r in all_records if r.record_type.value == "vaccination"]
            vaccinations.sort(key=lambda x: x.record_date, reverse=True)
            
            # === СДЕЛАННЫЕ ПРИВИВКИ ===
            
            vaccinations_done = []
            for vacc in vaccinations:
                next_due = None
                if vacc.next_visit_date:
                    next_due = vacc.next_visit_date.isoformat()
                else:
                    # По умолчанию через год
                    next_due = (vacc.record_date + timedelta(days=365)).isoformat()
                
                vaccinations_done.append({
                    "id": vacc.id,
                    "date": vacc.record_date.isoformat(),
                    "name": vacc.title,
                    "description": vacc.description,
                    "next_due": next_due,
                    "clinic": vacc.vet_clinic,
                    "vet_name": vacc.vet_name,
                    "cost": vacc.cost,
                })
            
            # === РЕКОМЕНДУЕМЫЕ ПРИВИВКИ ===
            
            # Справочник прививок по видам
            if pet.species.lower() in ["кошка", "cat", "кот"]:
                recommended = [
                    {
                        "name": "Бешенство",
                        "interval_days": 365,
                        "priority": "critical",
                        "age_first": "3 месяца",
                        "description": "Обязательная прививка, требуется для поездок"
                    },
                    {
                        "name": "Панлейкопения (кошачья чумка)",
                        "interval_days": 365,
                        "priority": "critical",
                        "age_first": "2 месяца",
                        "description": "Опасное вирусное заболевание"
                    },
                    {
                        "name": "Калицивироз",
                        "interval_days": 365,
                        "priority": "high",
                        "age_first": "2 месяца",
                        "description": "Вирусная инфекция дыхательных путей"
                    },
                    {
                        "name": "Ринотрахеит",
                        "interval_days": 365,
                        "priority": "high",
                        "age_first": "2 месяца",
                        "description": "Герпесвирусная инфекция"
                    },
                    {
                        "name": "Хламидиоз",
                        "interval_days": 365,
                        "priority": "medium",
                        "age_first": "2 месяца",
                        "description": "Бактериальная инфекция (опционально)"
                    },
                ]
            elif pet.species.lower() in ["собака", "dog", "пёс"]:
                recommended = [
                    {
                        "name": "Бешенство",
                        "interval_days": 365,
                        "priority": "critical",
                        "age_first": "3 месяца",
                        "description": "Обязательная прививка"
                    },
                    {
                        "name": "Чума плотоядных",
                        "interval_days": 365,
                        "priority": "critical",
                        "age_first": "2 месяца",
                        "description": "Смертельно опасное заболевание"
                    },
                    {
                        "name": "Парвовирусный энтерит",
                        "interval_days": 365,
                        "priority": "critical",
                        "age_first": "2 месяца",
                        "description": "Опасное кишечное заболевание"
                    },
                    {
                        "name": "Аденовироз (гепатит)",
                        "interval_days": 365,
                        "priority": "high",
                        "age_first": "2 месяца",
                        "description": "Инфекция печени и дыхательных путей"
                    },
                    {
                        "name": "Лептоспироз",
                        "interval_days": 365,
                        "priority": "high",
                        "age_first": "2 месяца",
                        "description": "Бактериальная инфекция (опасна для человека)"
                    },
                    {
                        "name": "Парагрипп",
                        "interval_days": 365,
                        "priority": "medium",
                        "age_first": "2 месяца",
                        "description": "Вирус дыхательных путей"
                    },
                ]
            else:
                # Для других животных - базовые прививки
                recommended = [
                    {
                        "name": "Бешенство",
                        "interval_days": 365,
                        "priority": "critical",
                        "age_first": "3 месяца",
                        "description": "Обязательная прививка"
                    },
                ]
            
            # === ПРОСРОЧЕННЫЕ ПРИВИВКИ ===
            
            overdue = []
            today = date.today()
            
            for rec in recommended:
                # Ищем последнюю прививку этого типа
                matching = [v for v in vaccinations if rec["name"].lower() in v.title.lower()]
                
                if matching:
                    last_vacc = matching[0]
                    next_due_date = last_vacc.record_date + timedelta(days=rec["interval_days"])
                    
                    if next_due_date < today:
                        overdue_days = (today - next_due_date).days
                        overdue.append({
                            "name": rec["name"],
                            "last_date": last_vacc.record_date.isoformat(),
                            "next_due_date": next_due_date.isoformat(),
                            "overdue_days": overdue_days,
                            "priority": rec["priority"],
                            "urgency": "критически" if overdue_days > 60 else "умеренно" if overdue_days > 30 else "незначительно"
                        })
            
            # === НУЖНЫЕ ПРИВИВКИ (ещё не сделанные) ===
            
            vaccinations_needed = []
            for rec in recommended:
                has_vacc = any(rec["name"].lower() in v.title.lower() for v in vaccinations)
                
                if not has_vacc:
                    # Определяем рекомендуемый возраст
                    age_rec = rec["age_first"]
                    if pet.age_years and pet.age_years >= 1:
                        age_rec = "как можно скорее"
                    
                    vaccinations_needed.append({
                        "name": rec["name"],
                        "recommended_age": age_rec,
                        "priority": rec["priority"],
                        "description": rec["description"],
                    })
            
            # === РЕКОМЕНДАЦИИ ===
            
            recommendations = []
            
            if overdue:
                critical_overdue = [v for v in overdue if v["priority"] == "critical"]
                if critical_overdue:
                    recommendations.append(f"⚠️ СРОЧНО: {len(critical_overdue)} критически важных прививок просрочены!")
            
            if vaccinations_needed:
                critical_needed = [v for v in vaccinations_needed if v["priority"] == "critical"]
                if critical_needed:
                    recommendations.append(f"⚠️ Не сделано {len(critical_needed)} критически важных прививок")
            
            if pet.age_years and pet.age_years < 1:
                recommendations.append("Котятам/щенкам требуется серия прививок с интервалом 3-4 недели")
            
            if not vaccinations:
                recommendations.append("У питомца нет записей о прививках. Обратитесь к ветеринару для составления графика вакцинации.")
            
            recommendations.append("Прививки - лучшая защита от опасных инфекций")
            recommendations.append("Перед вакцинацией необходима дегельминтизация (за 10-14 дней)")
            
            # === ФОРМИРУЕМ РЕЗУЛЬТАТ ===
            
            result = {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "pet": {
                    "name": pet.name,
                    "species": pet.species,
                    "breed": pet.breed,
                    "age_years": pet.age_years,
                },
                "summary": {
                    "total_vaccinations_done": len(vaccinations),
                    "vaccinations_needed_count": len(vaccinations_needed),
                    "overdue_count": len(overdue),
                },
                "vaccinations_done": vaccinations_done,
                "vaccinations_needed": vaccinations_needed,
                "overdue": overdue,
                "recommended_schedule": recommended,
                "recommendations": recommendations,
            }
            
            logger.info(f"Checked vaccinations for {pet.name}: done={len(vaccinations_done)}, overdue={len(overdue)}, needed={len(vaccinations_needed)}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to check vaccination schedule: {e}")
            return json.dumps({
                "error": str(e),
                "pet_name": pet_name
            }, ensure_ascii=False)


def create_health_nutrition_agent(
    pet_service: PetService,
    health_service: HealthRecordService,
    llm,
    name: str = "health_nutrition",
):
    """Создать агента для анализа здоровья и питания питомцев
    
    Args:
        pet_service: Сервис для работы с питомцами
        health_service: Сервис для работы с медицинскими записями
        llm: Языковая модель
        name: Имя агента (для supervisor handoff)
    
    Returns:
        Compiled ReAct agent
    """
    # Создаём экземпляр класса с инжектированными сервисами
    tools_instance = HealthNutritionTools(pet_service, health_service)
    
    # Собираем все методы, помеченные как @tool
    tools = [
        tools_instance.analyze_health_records,
        tools_instance.calculate_daily_nutrition,
        tools_instance.analyze_food_ingredients,
        tools_instance.check_vaccination_schedule,
    ]
    
    prompt = (
        "Ты - эксперт по здоровью и питанию домашних животных.\n\n"
        "Твои возможности:\n"
        "- Анализ медицинских записей питомца (история болезней, тренды веса/температуры)\n"
        "- Расчёт суточной нормы питания (калории, белки, жиры, углеводы)\n"
        "- Анализ состава корма (оценка качества ингредиентов)\n"
        "- Проверка графика прививок (сделанные, просроченные, необходимые)\n\n"
        "ВАЖНО: Ты работаешь ТОЛЬКО с домашними животными (кошки, собаки, грызуны и т.д.).\n"
        "Если вопрос НЕ о питомце - сообщи пользователю, что ты специализируешься только на животных.\n\n"
        "Все tools возвращают детальный JSON с анализом, рекомендациями и предупреждениями.\n"
        "Используй эту информацию для формирования профессионального ответа.\n\n"
        "Всегда напоминай: рекомендации носят общий характер, точный диагноз и лечение - только у ветеринара!"
    )
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        name=name,
        prompt=prompt,
    )
    
    logger.info(f"Created HealthNutritionAgent '{name}' with {len(tools)} tools")
    return agent