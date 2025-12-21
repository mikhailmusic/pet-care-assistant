from __future__ import annotations

from typing import Optional, Annotated, List, Literal
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from loguru import logger
import json

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from app.integrations.google_calendar_client import GoogleCalendarClient
from app.config import settings


def _parse_datetime(dt_str: str, user_timezone: str = settings.DEFAULT_TIMEZONE) -> datetime:
    if isinstance(dt_str, datetime):
        if dt_str.tzinfo is None:
            try:
                tz = ZoneInfo(user_timezone)
            except Exception:
                tz = timezone.utc
            return dt_str.replace(tzinfo=tz)
        return dt_str
    
    dt_str = dt_str.replace("Z", "+00:00")
    
    try:
        dt = datetime.fromisoformat(dt_str)
    except ValueError as e:
        raise ValueError(f"Invalid datetime format: {dt_str}") from e
    
    if dt.tzinfo is None:
        try:
            tz = ZoneInfo(user_timezone)
        except Exception:
            tz = timezone.utc
        dt = dt.replace(tzinfo=tz)
    
    return dt


def _get_rfc3339_time(dt: datetime) -> str:
    """Получить время в RFC3339 формате"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.isoformat()


def _parse_recurrence(recurrence_str: Optional[str]) -> Optional[List[str]]:
    if not recurrence_str:
        return None
    
    recurrence_map = {
        "ежедневно": "RRULE:FREQ=DAILY",
        "еженедельно": "RRULE:FREQ=WEEKLY",
        "ежемесячно": "RRULE:FREQ=MONTHLY",
        "ежегодно": "RRULE:FREQ=YEARLY",
    }
    
    rule = recurrence_map.get(recurrence_str.lower())
    return [rule] if rule else None


class CalendarTools:
    
    def __init__(self, calendar_client: GoogleCalendarClient, user_timezone: str = settings.DEFAULT_TIMEZONE):
        self.calendar_client = calendar_client
        self.user_timezone = user_timezone
    
    @tool
    async def create_calendar_event(
        self,
        state: Annotated[dict, InjectedState],
        title: str,
        start_datetime: str,
        end_datetime: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        recurrence: Optional[Literal["ежедневно", "еженедельно", "ежемесячно", "ежегодно"]] = None,
        attendees: Optional[List[str]] = None,
        reminder_minutes: Optional[List[int]] = None,
    ) -> str:
        """Создать событие в Google Calendar.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            title: Название события (обязательно)
            start_datetime: Дата и время начала в формате YYYY-MM-DDTHH:MM:SS
            end_datetime: Дата и время окончания (опционально, по умолчанию +1 час)
            description: Описание события
            location: Место проведения
            recurrence: Повторение события (ежедневно/еженедельно/ежемесячно/ежегодно)
            attendees: Список email участников
            reminder_minutes: Список минут до начала для уведомлений (например, [5, 30, 60])
        
        Returns:
            JSON с информацией о созданном событии
        """
        try:
            # Парсим start_datetime
            start_dt = _parse_datetime(start_datetime, self.user_timezone)
            
            # Парсим end_datetime или +1 час
            if end_datetime:
                end_dt = _parse_datetime(end_datetime, self.user_timezone)
            else:
                end_dt = start_dt + timedelta(hours=1)
            
            # Преобразуем recurrence в RRULE
            recurrence_rules = _parse_recurrence(recurrence)

            # Настраиваем напоминания
            reminders = None
            if reminder_minutes:
                reminders = {
                    "useDefault": False,
                    "overrides": [
                        {"method": "popup", "minutes": minutes}
                        for minutes in reminder_minutes
                    ]
                }
            
            # Создаём событие
            event = self.calendar_client.create_event(
                summary=title,
                start_time=start_dt.isoformat(),
                end_time=end_dt.isoformat(),
                description=description,
                location=location,
                timezone=self.user_timezone,
                recurrence=recurrence_rules,
                attendees=attendees,
                send_updates="all" if attendees else "none",
                reminders=reminders,
            )
            
            result = {
                "success": True,
                "event_id": event.get("id"),
                "title": title,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "link": event.get("htmlLink"),
            }
            
            logger.info(f"Created calendar event '{title}' at {start_dt.isoformat()}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, ensure_ascii=False)

    @tool
    async def list_calendar_events(
        self,
        state: Annotated[dict, InjectedState],
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        query: Optional[str] = None,
        max_results: int = 10,
    ) -> str:
        """Получить список событий из Google Calendar.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            time_min: Начало периода в формате YYYY-MM-DDTHH:MM:SS (по умолчанию: сейчас)
            time_max: Конец периода в формате YYYY-MM-DDTHH:MM:SS (по умолчанию: +30 дней)
            query: Текст для поиска в названии или описании события
            max_results: Максимальное количество результатов
        
        Returns:
            JSON со списком найденных событий
        """
        try:
            # Определяем временной диапазон
            if not time_min:
                dt_min = datetime.now(timezone.utc)
            else:
                dt_min = _parse_datetime(time_min, self.user_timezone)

            if not time_max:
                # Умная логика: если time_min указан в начале дня (00:00:00),
                # то time_max = конец того же дня, иначе +30 дней
                if dt_min.hour == 0 and dt_min.minute == 0 and dt_min.second == 0:
                    # Запрос событий на конкретный день
                    dt_max = dt_min.replace(hour=23, minute=59, second=59)
                else:
                    # Общий поиск вперёд
                    dt_max = datetime.now(timezone.utc) + timedelta(days=30)
            else:
                dt_max = _parse_datetime(time_max, self.user_timezone)
            
            # Получаем события
            events = self.calendar_client.list_events(
                time_min=_get_rfc3339_time(dt_min),
                time_max=_get_rfc3339_time(dt_max),
                max_results=max_results * 2,  # Берём с запасом для фильтрации
                single_events=True
            )
            
            # Фильтруем по query
            if query:
                query_lower = query.lower()
                events = [
                    e for e in events
                    if query_lower in e.get("summary", "").lower() or
                       query_lower in e.get("description", "").lower()
                ]
            
            # Ограничиваем результаты
            events = events[:max_results]
            
            if not events:
                return json.dumps({
                    "success": True,
                    "count": 0,
                    "events": []
                }, ensure_ascii=False)
            
            # Форматируем результаты
            formatted_events = []
            for event in events:
                title = event.get("summary", "Без названия")
                start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date", ""))
                end = event.get("end", {}).get("dateTime", event.get("end", {}).get("date", ""))
                
                formatted_events.append({
                    "id": event.get("id"),
                    "title": title,
                    "start": start,
                    "end": end,
                    "description": event.get("description"),
                    "location": event.get("location"),
                    "link": event.get("htmlLink"),
                })
            
            result = {
                "success": True,
                "count": len(formatted_events),
                "events": formatted_events
            }
            
            logger.info(f"Found {len(events)} calendar events")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to list events: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, ensure_ascii=False)

    @tool
    async def update_calendar_event(
        self,
        state: Annotated[dict, InjectedState],
        search_query: str,
        new_title: Optional[str] = None,
        new_start_datetime: Optional[str] = None,
        new_end_datetime: Optional[str] = None,
        new_description: Optional[str] = None,
        new_location: Optional[str] = None,
        new_attendees: Optional[List[str]] = None,
    ) -> str:
        """Обновить существующее событие в Google Calendar.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            search_query: Текст для поиска события (название или описание)
            new_title: Новое название события
            new_start_datetime: Новая дата/время начала в формате YYYY-MM-DDTHH:MM:SS
            new_end_datetime: Новая дата/время окончания
            new_description: Новое описание
            new_location: Новое место
            new_attendees: Новый список участников
        
        Returns:
            JSON с результатом обновления
        """
        try:
            # Ищем событие
            now = datetime.now(timezone.utc)
            events = self.calendar_client.list_events(
                time_min=_get_rfc3339_time(now - timedelta(days=90)),
                time_max=_get_rfc3339_time(now + timedelta(days=180)),
                max_results=100,
                single_events=True
            )
            
            # Фильтруем по search_query
            query_lower = search_query.lower()
            matching_events = [
                e for e in events
                if query_lower in e.get("summary", "").lower() or
                   query_lower in e.get("description", "").lower()
            ]
            
            if not matching_events:
                return json.dumps({
                    "success": False,
                    "error": f"Событие '{search_query}' не найдено"
                }, ensure_ascii=False)
            
            if len(matching_events) > 1:
                titles = [e.get("summary", "Без названия") for e in matching_events[:3]]
                return json.dumps({
                    "success": False,
                    "error": f"Найдено несколько событий: {', '.join(titles)}. Уточните запрос."
                }, ensure_ascii=False)
            
            event_id = matching_events[0].get("id")
            
            # Парсим новые даты
            start_time = None
            end_time = None
            
            if new_start_datetime:
                start_dt = _parse_datetime(new_start_datetime, self.user_timezone)
                start_time = start_dt.isoformat()
            
            if new_end_datetime:
                end_dt = _parse_datetime(new_end_datetime, self.user_timezone)
                end_time = end_dt.isoformat()
            
            # Обновляем событие
            updated_event = self.calendar_client.update_event(
                event_id=event_id,
                summary=new_title,
                start_time=start_time,
                end_time=end_time,
                description=new_description,
                location=new_location,
                attendees=new_attendees,
                timezone=self.user_timezone,
            )
            
            result = {
                "success": True,
                "event_id": event_id,
                "title": matching_events[0].get("summary"),
                "link": updated_event.get("htmlLink"),
            }
            
            logger.info(f"Updated calendar event: {event_id}")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to update event: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, ensure_ascii=False)

    @tool
    async def delete_calendar_event(
        self,
        state: Annotated[dict, InjectedState],
        search_query: str,
    ) -> str:
        """Удалить событие из Google Calendar.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            search_query: Текст для поиска события (название или описание)
        
        Returns:
            JSON с результатом удаления
        """
        try:
            # Ищем событие
            now = datetime.now(timezone.utc)
            events = self.calendar_client.list_events(
                time_min=_get_rfc3339_time(now - timedelta(days=90)),
                time_max=_get_rfc3339_time(now + timedelta(days=180)),
                max_results=100,
                single_events=True
            )
            
            # Фильтруем по search_query
            query_lower = search_query.lower()
            matching_events = [
                e for e in events
                if query_lower in e.get("summary", "").lower() or
                   query_lower in e.get("description", "").lower()
            ]
            
            if not matching_events:
                return json.dumps({
                    "success": False,
                    "error": f"Событие '{search_query}' не найдено"
                }, ensure_ascii=False)
            
            if len(matching_events) > 1:
                titles = [e.get("summary", "Без названия") for e in matching_events[:3]]
                return json.dumps({
                    "success": False,
                    "error": f"Найдено несколько событий: {', '.join(titles)}. Уточните запрос."
                }, ensure_ascii=False)
            
            event_id = matching_events[0].get("id")
            event_title = matching_events[0].get("summary", "Без названия")
            
            # Удаляем событие
            success = self.calendar_client.delete_event(event_id=event_id)
            
            if success:
                result = {
                    "success": True,
                    "event_id": event_id,
                    "title": event_title,
                }
                logger.info(f"Deleted calendar event: {event_id}")
                return json.dumps(result, ensure_ascii=False, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "error": "Не удалось удалить событие"
                }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Failed to delete event: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, ensure_ascii=False)

    @tool
    async def check_calendar_availability(
        self,
        state: Annotated[dict, InjectedState],
        time_min: str,
        time_max: str,
    ) -> str:
        """Проверить свободное время в календаре.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            time_min: Начало периода в формате YYYY-MM-DDTHH:MM:SS
            time_max: Конец периода в формате YYYY-MM-DDTHH:MM:SS
        
        Returns:
            JSON с информацией о занятости
        """
        try:
            # Парсим даты
            dt_min = _parse_datetime(time_min, self.user_timezone)
            dt_max = _parse_datetime(time_max, self.user_timezone)
            
            # Проверяем занятость
            freebusy = self.calendar_client.check_freebusy(
                calendars=["primary"],
                time_min=_get_rfc3339_time(dt_min),
                time_max=_get_rfc3339_time(dt_max),
                timezone=self.user_timezone
            )
            
            if not freebusy:
                return json.dumps({
                    "success": False,
                    "error": "Не удалось проверить занятость"
                }, ensure_ascii=False)
            
            # Извлекаем занятые промежутки
            busy_periods = freebusy.get("calendars", {}).get("primary", {}).get("busy", [])
            
            result = {
                "success": True,
                "period": {
                    "start": time_min,
                    "end": time_max,
                },
                "is_free": len(busy_periods) == 0,
                "busy_count": len(busy_periods),
                "busy_periods": [
                    {
                        "start": period.get("start"),
                        "end": period.get("end"),
                    }
                    for period in busy_periods
                ]
            }
            
            logger.info(f"Checked availability: {len(busy_periods)} busy periods")
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to check availability: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            }, ensure_ascii=False)

def create_calendar_agent(
    calendar_client: GoogleCalendarClient,
    user_timezone: str,
    llm,
    name: str = "calendar",
):
    """Создать агента для работы с Google Calendar
    
    Args:
        calendar_client: Клиент Google Calendar с настроенными credentials
        user_timezone: Часовой пояс пользователя
        llm: Языковая модель
        name: Имя агента (для supervisor handoff)
    
    Returns:
        Compiled ReAct agent
    """
    tools_instance = CalendarTools(calendar_client, user_timezone)
    
    tools = [
        tools_instance.create_calendar_event,
        tools_instance.list_calendar_events,
        tools_instance.update_calendar_event,
        tools_instance.delete_calendar_event,
        tools_instance.check_calendar_availability,
    ]
    
    now = datetime.now()
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    
    prompt = f"""Ты - помощник по работе с Google Calendar.

Текущая дата и время: {now.strftime("%Y-%m-%d %H:%M")} ({now.strftime("%A")})
Часовой пояс пользователя: {user_timezone}

Твои возможности:
- Создание событий (create_calendar_event)
- Поиск событий (list_calendar_events)
- Обновление событий (update_calendar_event)
- Удаление событий (delete_calendar_event)
- Проверка занятости (check_calendar_availability)

**ВАЖНО о уведомлениях (напоминаниях):**
- Фразы "напомни за X минут", "за X минут до начала" означают добавить reminder к событию
- НЕ создавай отдельные события для уведомлений!
- Используй параметр reminder_minutes в create_calendar_event
- Можно указать несколько уведомлений: reminder_minutes=[5, 30, 60]

**Правила обработки дат:**
- "завтра" → {tomorrow}
- "послезавтра" → {day_after}
- "через неделю" → {(now + timedelta(days=7)).strftime("%Y-%m-%d")}
- Если время не указано, используй 10:00:00
- Формат datetime: YYYY-MM-DDTHH:MM:SS

**КРИТИЧНО для list_calendar_events:**
При поиске событий на конкретный день ВСЕГДА указывай оба параметра:
- time_min: начало дня (00:00:00)
- time_max: конец дня (23:59:59)

Примеры:
- "события на завтра" → list_calendar_events(time_min="{tomorrow}T00:00:00", time_max="{tomorrow}T23:59:59")
- "что у меня послезавтра" → list_calendar_events(time_min="{day_after}T00:00:00", time_max="{day_after}T23:59:59")

Все tools возвращают JSON - используй его для формирования ответа пользователю.
"""
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        name=name,
        prompt=prompt,
    )
    
    logger.info(f"Created CalendarAgent '{name}' with {len(tools)} tools")
    return agent