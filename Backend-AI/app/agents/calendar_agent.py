from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from loguru import logger
from contextvars import ContextVar

from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.integrations.google_calendar_client import GoogleCalendarClient
from app.utils.exceptions import GoogleCalendarException
from app.integrations.gigachat_client import GigaChatClient
from app.config import settings


@dataclass
class CalendarContext:
    user_id: int
    calendar_client: GoogleCalendarClient
    user_timezone: str = "UTC"
    current_pet_name: str = ""


_calendar_context: ContextVar[Optional[CalendarContext]] = ContextVar('_calendar_context', default=None)


def _get_context() -> CalendarContext:
    """Get the current calendar context from ContextVar"""
    ctx = _calendar_context.get()
    if ctx is None:
        raise RuntimeError("Calendar context not set. This should not happen.")
    return ctx

# ============================================================================
# TOOLS
# ============================================================================

@tool
async def create_calendar_event(
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
        title: Название события (обязательно)
        start_datetime: Дата и время начала в формате YYYY-MM-DDTHH:MM:SS
        end_datetime: Дата и время окончания (опционально, по умолчанию +1 час)
        description: Описание события
        location: Место проведения
        recurrence: Повторение события
        attendees: Список email участников
        reminder_minutes: Список минут до начала для уведомлений (например, [5, 30, 60])
    
    Returns:
        Информация о созданном событии
    """
    try:
        ctx = _get_context()
        
        # Парсим start_datetime
        start_dt = _parse_datetime(start_datetime, ctx.user_timezone)
        
        # Парсим end_datetime или +1 час
        if end_datetime:
            end_dt = _parse_datetime(end_datetime, ctx.user_timezone)
        else:
            end_dt = start_dt + timedelta(hours=1)
        
        # Преобразуем recurrence в RRULE
        recurrence_rules = _parse_recurrence(recurrence)

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
        event = ctx.calendar_client.create_event(
            summary=title,
            start_time=start_dt.isoformat(),
            end_time=end_dt.isoformat(),
            description=description,
            location=location,
            timezone=ctx.user_timezone,
            recurrence=recurrence_rules,
            attendees=attendees,
            send_updates="all" if attendees else "none",
            reminders=reminders,
        )
        
        logger.info(f"Created event '{title}' at {start_dt.isoformat()}")
        return f"✅ Событие '{title}' создано на {start_dt.strftime('%d.%m.%Y %H:%M')}"
        
    except Exception as e:
        logger.error(f"Failed to create event: {e}")
        return f"❌ Ошибка создания события: {str(e)}"


@tool
async def list_calendar_events(
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    query: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Получить список событий из Google Calendar.
    
    Args:
        time_min: Начало периода в формате YYYY-MM-DDTHH:MM:SS (по умолчанию: сейчас)
        time_max: Конец периода в формате YYYY-MM-DDTHH:MM:SS (по умолчанию: +30 дней)
        query: Текст для поиска в названии или описании события
        max_results: Максимальное количество результатов
    
    Returns:
        Список найденных событий
    """
    try:
        ctx = _get_context()
        
        # Определяем временной диапазон
        if not time_min:
            dt_min = datetime.now(timezone.utc)
        else:
            dt_min = _parse_datetime(time_min, ctx.user_timezone)
        
        if not time_max:
            dt_max = datetime.now(timezone.utc) + timedelta(days=30)
        else:
            dt_max = _parse_datetime(time_max, ctx.user_timezone)
        
        # Получаем события
        events = ctx.calendar_client.list_events(
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
            return "Событий не найдено в указанном периоде."
        
        # Форматируем результат
        result = f"Найдено событий: {len(events)}\n\n"
        for i, event in enumerate(events, 1):
            title = event.get("summary", "Без названия")
            start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date", ""))
            
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                start_str = start_dt.strftime("%d.%m.%Y %H:%M")
            except:
                start_str = start
            
            result += f"{i}. {title} - {start_str}\n"
            
            if desc := event.get("description"):
                result += f"   📝 {desc[:50]}...\n" if len(desc) > 50 else f"   📝 {desc}\n"
        
        logger.info(f"Found {len(events)} events")
        return result
        
    except Exception as e:
        logger.error(f"Failed to list events: {e}")
        return f"❌ Ошибка получения событий: {str(e)}"


@tool
async def update_calendar_event(
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
        search_query: Текст для поиска события (название или описание)
        new_title: Новое название события
        new_start_datetime: Новая дата/время начала в формате YYYY-MM-DDTHH:MM:SS
        new_end_datetime: Новая дата/время окончания
        new_description: Новое описание
        new_location: Новое место
        new_attendees: Новый список участников
    
    Returns:
        Результат обновления
    """
    try:
        ctx = _get_context()
        
        # Ищем событие
        now = datetime.now(timezone.utc)
        events = ctx.calendar_client.list_events(
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
            return f"❌ Событие '{search_query}' не найдено"
        
        if len(matching_events) > 1:
            titles = [e.get("summary", "Без названия") for e in matching_events[:3]]
            return f"❌ Найдено несколько событий: {', '.join(titles)}. Уточните запрос."
        
        event_id = matching_events[0].get("id")
        
        # Парсим новые даты
        start_time = None
        end_time = None
        
        if new_start_datetime:
            start_dt = _parse_datetime(new_start_datetime, ctx.user_timezone)
            start_time = start_dt.isoformat()
        
        if new_end_datetime:
            end_dt = _parse_datetime(new_end_datetime, ctx.user_timezone)
            end_time = end_dt.isoformat()
        
        # Обновляем событие
        updated_event = ctx.calendar_client.update_event(
            event_id=event_id,
            summary=new_title,
            start_time=start_time,
            end_time=end_time,
            description=new_description,
            location=new_location,
            attendees=new_attendees,
            timezone=ctx.user_timezone,
        )
        
        logger.info(f"Updated event: {event_id}")
        return f"✅ Событие '{matching_events[0].get('summary')}' обновлено"
        
    except Exception as e:
        logger.error(f"Failed to update event: {e}")
        return f"❌ Ошибка обновления события: {str(e)}"


@tool
async def delete_calendar_event(
    search_query: str,
) -> str:
    """Удалить событие из Google Calendar.
    
    Args:
        search_query: Текст для поиска события (название или описание)
    
    Returns:
        Результат удаления
    """
    try:
        ctx = _get_context()
        
        # Ищем событие
        now = datetime.now(timezone.utc)
        events = ctx.calendar_client.list_events(
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
            return f"❌ Событие '{search_query}' не найдено"
        
        if len(matching_events) > 1:
            titles = [e.get("summary", "Без названия") for e in matching_events[:3]]
            return f"❌ Найдено несколько событий: {', '.join(titles)}. Уточните запрос."
        
        event_id = matching_events[0].get("id")
        event_title = matching_events[0].get("summary", "Без названия")
        
        # Удаляем событие
        success = ctx.calendar_client.delete_event(event_id=event_id)
        
        if success:
            logger.info(f"Deleted event: {event_id}")
            return f"✅ Событие '{event_title}' удалено"
        else:
            return "❌ Не удалось удалить событие"
        
    except Exception as e:
        logger.error(f"Failed to delete event: {e}")
        return f"❌ Ошибка удаления события: {str(e)}"


@tool
async def check_calendar_availability(
    time_min: str,
    time_max: str,
) -> str:
    """Проверить свободное время в календаре.
    
    Args:
        time_min: Начало периода в формате YYYY-MM-DDTHH:MM:SS
        time_max: Конец периода в формате YYYY-MM-DDTHH:MM:SS
    
    Returns:
        Информация о занятости
    """
    try:
        ctx = _get_context()
        
        # Парсим даты
        dt_min = _parse_datetime(time_min, ctx.user_timezone)
        dt_max = _parse_datetime(time_max, ctx.user_timezone)
        
        # Проверяем занятость
        freebusy = ctx.calendar_client.check_freebusy(
            calendars=["primary"],
            time_min=_get_rfc3339_time(dt_min),
            time_max=_get_rfc3339_time(dt_max),
            timezone=ctx.user_timezone
        )
        
        if not freebusy:
            return "❌ Не удалось проверить занятость"
        
        # Извлекаем занятые промежутки
        busy_periods = freebusy.get("calendars", {}).get("primary", {}).get("busy", [])
        
        if not busy_periods:
            period_str = f"{dt_min.strftime('%d.%m.%Y')} с {dt_min.strftime('%H:%M')} до {dt_max.strftime('%H:%M')}"
            return f"✅ В период {period_str} вы полностью свободны"
        
        # Форматируем занятые промежутки
        result = f"📅 Занято {len(busy_periods)} промежутков:\n\n"
        for i, period in enumerate(busy_periods[:10], 1):
            start = period.get("start", "")
            end = period.get("end", "")
            
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                result += f"{i}. {start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}\n"
            except:
                result += f"{i}. {start} - {end}\n"
        
        if len(busy_periods) > 10:
            result += f"\n... и ещё {len(busy_periods) - 10} промежутков"
        
        logger.info(f"Found {len(busy_periods)} busy periods")
        return result
        
    except Exception as e:
        logger.error(f"Failed to check availability: {e}")
        return f"❌ Ошибка проверки занятости: {str(e)}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _parse_datetime(dt_str: str, user_timezone: str = "UTC") -> datetime:
    """Парсинг datetime из строки"""
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
    """Преобразовать описание повторения в RRULE формат"""
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


# ============================================================================
# CALENDAR AGENT
# ============================================================================

class CalendarAgent:
    """Агент для работы с Google Calendar через LangChain tools"""
    
    def __init__(self, user_service, llm=None):
        """
        Args:
            user_service: Сервис для работы с пользователями
            llm: LLM для агента (по умолчанию ChatOpenAI)
        """
        self.user_service = user_service
        self.llm = llm or GigaChatClient().llm
        
        # Список инструментов
        self.tools = [
            create_calendar_event,
            list_calendar_events,
            update_calendar_event,
            delete_calendar_event,
            check_calendar_availability,
        ]
        
        logger.info("CalendarAgent initialized with tools")
    
    async def process(
        self,
        user_id: int,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Обработать сообщение пользователя
        
        Args:
            user_id: ID пользователя
            user_message: Сообщение пользователя
            context: Контекст (user_timezone, current_pet_name)
        
        Returns:
            Ответ агента
        """
        context = context or {}
        
        try:
            # Получаем credentials
            creds_json = await self.user_service.get_google_credentials(user_id)
            
            if not creds_json:
                return "❌ Google Calendar не подключен. Подключите его в настройках."
            
            # Инициализируем клиент
            calendar_client = GoogleCalendarClient()
            
            try:
                calendar_client.set_credentials_from_json(creds_json)
            except Exception as e:
                logger.error(f"Invalid credentials for user {user_id}: {e}")
                return "❌ Токен Google устарел. Переавторизуйтесь в Google Calendar."
            
            # Создаём контекст для tools
            tool_context = CalendarContext(
                user_id=user_id,
                calendar_client=calendar_client,
                user_timezone=context.get("user_timezone", "UTC"),
                current_pet_name=context.get("current_pet_name", "")
            )
            
            # Формируем system prompt с текущей датой
            now = datetime.now()
            tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            day_after = (now + timedelta(days=2)).strftime("%Y-%m-%d")
            system_prompt = f"""Ты - помощник по работе с Google Calendar.

Текущая дата и время: {now.strftime("%Y-%m-%d %H:%M")} ({now.strftime("%A")})
Часовой пояс пользователя: {tool_context.user_timezone}
{f'Питомец: {tool_context.current_pet_name}' if tool_context.current_pet_name else ''}

Используй доступные инструменты для работы с календарём.

**ВАЖНО о уведомлениях (напоминаниях):**
- Фразы "напомни за X минут", "за X минут до начала" означают добавить reminder к событию
- НЕ создавай отдельные события для уведомлений!
- Используй параметр reminder_minutes в create_calendar_event
- Можно указать несколько уведомлений: reminder_minutes=[5, 30, 60]


Правила обработки дат:
- "завтра" → {(now + timedelta(days=1)).strftime("%Y-%m-%d")}
- "послезавтра" → {(now + timedelta(days=2)).strftime("%Y-%m-%d")}
- "через неделю" → {(now + timedelta(days=7)).strftime("%Y-%m-%d")}
- Если время не указано, используй 10:00:00
- Формат datetime: YYYY-MM-DDTHH:MM:SS


**Примеры:**

1. "Встреча завтра в 15:00, напомни за 20 и за 5 минут" →
   create_calendar_event(title="Встреча", start_datetime="{tomorrow}T15:00:00", reminder_minutes=[20, 5])

2. "Ветеринар послезавтра в 10:00 на 30 минут, напомни за 50 минут" →
   create_calendar_event(title="Ветеринар", start_datetime="{day_after}T10:00:00", 
                         end_datetime="{day_after}T10:30:00", reminder_minutes=[50])

При создании событий всегда используй имя питомца в названии, если оно указано. Добавляй другие параметры события, если они присутствуют
"""

            
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
            
            token = _calendar_context.set(tool_context)
            
            try:
                # Вызываем агента
                result = await agent_executor.ainvoke({"input": user_message})
            finally:
                # Сбрасываем контекст
                _calendar_context.reset(token)
            
            # Сохраняем обновлённые credentials
            try:
                new_creds_json = calendar_client.get_credentials_json()
                if new_creds_json != creds_json:
                    await self.user_service.add_google_credentials(user_id, new_creds_json)
                    logger.info(f"Refreshed Google credentials for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to save refreshed credentials: {e}")
            
            return result.get("output", "Обработан запрос календаря")
            
        except GoogleCalendarException as e:
            logger.error(f"Google Calendar error for user {user_id}: {e}")
            return f"❌ Ошибка Google Calendar: {str(e)}"
        except Exception as e:
            logger.exception(f"CalendarAgent unexpected error for user {user_id}")
            return "❌ Внутренняя ошибка календаря"
