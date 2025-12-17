import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow, Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger
from app.utils.exceptions import GoogleCalendarException
from app.config import settings

SCOPES = ['https://www.googleapis.com/auth/calendar']
REDIRECT_URI = "http://localhost:8000/auth/callback"

class GoogleCalendarClient:
    
    def __init__(self, credentials_file: Optional[str] = None, 
                 token_file: Optional[str] = None):
        """
        Инициализация клиента
        
        Args:
            credentials_file: Путь к файлу с OAuth credentials
            token_file: Путь к файлу для хранения токенов
        """
        self.credentials_file = credentials_file or settings.GOOGLE_CALENDAR_CREDENTIALS_FILE
        self.token_file = token_file or settings.GOOGLE_CALENDAR_TOKEN_FILE
        self.creds = None
        self.service = None
    
    def authenticate(self, use_local_server: bool = True) -> None:
        """
        
        Args:
            use_local_server: Использовать локальный сервер для OAuth (True) 
                            или консольный ввод (False)
        """
        # Загрузить существующие токены
        if os.path.exists(self.token_file):
            self.creds = Credentials.from_authorized_user_file(
                self.token_file, SCOPES
            )
        
        # Если токенов нет или они невалидны
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                if use_local_server:
                    self.creds = flow.run_local_server(
                        port=8080,
                        access_type='offline',
                        prompt='consent'
                    )                
                else:
                    self.creds = flow.run_console(
                        access_type='offline',
                        prompt='consent'
                    )  
            
            # Сохранить токены
            with open(self.token_file, 'w') as token:
                token.write(self.creds.to_json())
        
        self.service = build('calendar', 'v3', credentials=self.creds)
    
    def set_credentials_from_json(self, creds_json: str) -> None:
        info = json.loads(creds_json)
        self.creds = Credentials.from_authorized_user_info(info, SCOPES)
        if self.creds.expired and self.creds.refresh_token:
            self.creds.refresh(Request())
        self.service = build('calendar', 'v3', credentials=self.creds)

    def get_credentials_json(self) -> str:
        if not self.creds:
            raise RuntimeError("Credentials are not set")
        return self.creds.to_json()

    def _build_web_flow(self, redirect_uri: str) -> Flow:
        return Flow.from_client_secrets_file(
            self.credentials_file,
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )

    def get_authorization_url(self, redirect_uri: str = REDIRECT_URI, state: Optional[str] = None) -> tuple[str, Optional[str]]:
        """
        Формирует URL для OAuth авторизации (веб-поток).

        Returns: (auth_url, state)
        """
        try:
            flow = self._build_web_flow(redirect_uri)
            auth_url, flow_state = flow.authorization_url(
                access_type='offline',
                prompt='consent',
                include_granted_scopes='true',
                state=state
            )
            return auth_url, flow_state
        except Exception as exc:
            logger.error(f"Failed to build Google auth URL: {exc}")
            raise GoogleCalendarException("Не удалось сформировать ссылку для авторизации Google")

    def exchange_code_for_credentials(self, code: str, redirect_uri: str = REDIRECT_URI) -> str:
        """
        Обменивает authorization code (из redirect) на JSON с access/refresh токенами.

        ВАЖНО: redirect_uri должен точно совпадать с тем, что использовался при get_authorization_url()
        """
        try:
            logger.info(f"Exchanging code with redirect_uri: {redirect_uri}")
            flow = self._build_web_flow(redirect_uri)
            # Используем только code, не authorization_response
            flow.fetch_token(code=code)
            self.creds = flow.credentials
            # Не обновляем токен сразу после получения - он свежий
            self.service = build('calendar', 'v3', credentials=self.creds)
            return self.creds.to_json()
        except Exception as exc:
            logger.error(f"Failed to exchange Google auth code: {exc}")
            # Пробрасываем первичную причину, чтобы фронт/логи видели реальную ошибку (часто invalid_grant из-за redirect URI).
            raise GoogleCalendarException(f"Не удалось получить учетные данные Google: {exc}")
    
    def web_auth_url(self, redirect_uri: str = REDIRECT_URI, state: Optional[str] = None) -> str:
        """
        ?????>???????'?? URL ???>?? ?????+-?????'???????????????? (???>?? Flask/FastAPI)
        
        Args:
            redirect_uri: URL ???>?? ???????????????'?? ???????>?? ?????'????????????????
            state: ???????????????>?????<?? state ???>?? ?????%???'?< ???' CSRF
            
        Returns:
            URL ???>?? ?????????????????????>???????? ?????>???????????'???>??
        """
        auth_url, _ = self.get_authorization_url(redirect_uri=redirect_uri, state=state)
        return auth_url

    def web_auth_callback(self, authorization_response: str, redirect_uri: str = REDIRECT_URI) -> None:
        """
        Завершить веб-авторизацию и получить токены
        
        Args:
            authorization_response: Полный URL с кодом авторизации
            redirect_uri: URL для редиректа (тот же, что в web_auth_url)
        """
        flow = Flow.from_client_secrets_file(
            self.credentials_file,
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )
        flow.fetch_token(authorization_response=authorization_response)
        self.creds = flow.credentials
        
        with open(self.token_file, 'w') as token:
            token.write(self.creds.to_json())
        
        self.service = build('calendar', 'v3', credentials=self.creds)
    

    
    def list_events(self, calendar_id: str = 'primary', 
                   max_results: int = 10,
                   time_min: Optional[str] = None,
                   time_max: Optional[str] = None,
                   single_events: bool = True,
                   order_by: str = 'startTime') -> List[Dict[str, Any]]:
        """
        Получить список событий
        
        Args:
            calendar_id: ID календаря ('primary' для основного)
            max_results: Макс. кол-во событий
            time_min: Начало периода (RFC3339, напр. '2025-12-09T00:00:00Z')
            time_max: Конец периода
            single_events: Развернуть повторяющиеся события
            order_by: Сортировка ('startTime' или 'updated')
            
        Returns:
            Список событий
        """
        try:
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=single_events,
                orderBy=order_by
            ).execute()
            return events_result.get('items', [])
        except HttpError as error:
            logger.error(f'Google Calendar API error: {error}')
            if error.resp.status == 401:
                raise GoogleCalendarException("Токен истек или невалиден. Требуется повторная авторизация.")
            elif error.resp.status == 403:
                raise GoogleCalendarException("Недостаточно прав для доступа к календарю.")
            elif error.resp.status == 404:
                raise GoogleCalendarException("Календарь не найден.")
            else:
                raise GoogleCalendarException(f"Ошибка Google Calendar API: {error}")

    def list_all_events(self, 
                        max_results: int = 10,
                        time_min: Optional[str] = None,
                        time_max: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Получить события из всех календарей пользователя
        
        Returns:
            Dict с ключами - названиями календарей и значениями - списками событий
        """
        try:
            calendars = self.list_calendars()
            
            all_events = {}
            for cal in calendars:
                cal_id = cal['id']
                cal_name = cal.get('summary', cal_id)
                
                events = self.list_events(
                    calendar_id=cal_id,
                    max_results=max_results,
                    time_min=time_min,
                    time_max=time_max
                )
                
                all_events[cal_name] = events
            
            return all_events
            
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при получении всех событий")

    
    def get_event(self, event_id: str, 
                  calendar_id: str = 'primary') -> Optional[Dict[str, Any]]:
        """
        Получить одно событие по ID
        
        Args:
            event_id: ID события
            calendar_id: ID календаря
            
        Returns:
            Данные события или None
        """
        try:
            event = self.service.events().get(
                calendarId=calendar_id,
                eventId=event_id
            ).execute()
            return event
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при получении события")

    
    def create_event(self, summary: str, 
                    start_time: str,
                    end_time: str,
                    send_updates: str = 'none',
                    description: Optional[str] = None,
                    location: Optional[str] = None,
                    attendees: Optional[List[str]] = None,
                    timezone: str = 'UTC',
                    calendar_id: str = 'primary',
                    recurrence: Optional[List[str]] = None,
                    reminders: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Создать новое событие
        
        Args:
            summary: Название события
            start_time: Начало (ISO 8601: '2025-12-09T10:00:00')
            end_time: Конец
            description: Описание
            location: Место
            attendees: Список email участников
            timezone: Часовой пояс (напр. 'Europe/Moscow')
            calendar_id: ID календаря
            recurrence: Правила повторения (напр. ['RRULE:FREQ=DAILY;COUNT=5'])
            reminders: Напоминания, напр. {'useDefault': False, 
                      'overrides': [{'method': 'popup', 'minutes': 10}]}
            
        Returns:
            Созданное событие или None
        """
        event_body = {
            'summary': summary,
            'start': {
                'dateTime': start_time,
                'timeZone': timezone,
            },
            'end': {
                'dateTime': end_time,
                'timeZone': timezone,
            },
        }
        
        if description:
            event_body['description'] = description
        if location:
            event_body['location'] = location
        if attendees:
            event_body['attendees'] = [{'email': email} for email in attendees]
        if recurrence:
            event_body['recurrence'] = recurrence
        if reminders:
            event_body['reminders'] = reminders
        
        try:
            event = self.service.events().insert(
                calendarId=calendar_id,
                body=event_body,
                sendUpdates=send_updates
            ).execute()
            return event
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при создании события")

    
    def update_event(self, event_id: str,
                    summary: Optional[str] = None,
                    start_time: Optional[str] = None,
                    end_time: Optional[str] = None,
                    description: Optional[str] = None,
                    location: Optional[str] = None,
                    attendees: Optional[List[str]] = None,
                    timezone: str = 'UTC',
                    calendar_id: str = 'primary') -> Optional[Dict[str, Any]]:
        """
        Обновить существующее событие
        
        Args:
            event_id: ID события для обновления
            Остальные параметры как в create_event (если None, не меняются)
            
        Returns:
            Обновленное событие или None
        """
        try:
            # Получить текущее событие
            event = self.service.events().get(
                calendarId=calendar_id,
                eventId=event_id
            ).execute()
            
            # Обновить только переданные поля
            if summary:
                event['summary'] = summary
            if start_time:
                event['start'] = {'dateTime': start_time, 'timeZone': timezone}
            if end_time:
                event['end'] = {'dateTime': end_time, 'timeZone': timezone}
            if description:
                event['description'] = description
            if location:
                event['location'] = location
            if attendees:
                event['attendees'] = [{'email': email} for email in attendees]
            
            updated_event = self.service.events().update(
                calendarId=calendar_id,
                eventId=event_id,
                body=event,
                sendUpdates='all'
            ).execute()
            return updated_event
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при обновлении события")

    
    def delete_event(self, event_id: str, calendar_id: str = 'primary') -> bool:
        """        
        Args:
            event_id: ID события
            calendar_id: ID календаря
            
        Returns:
            True если успешно удалено
        """
        try:
            self.service.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
                sendUpdates='all'
            ).execute()
            return True
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при удалении события")

    
    def quick_add_event(self, text: str, calendar_id: str = 'primary') -> Optional[Dict[str, Any]]:
        """
        Быстрое создание события из текста (Google парсит сам)
        
        Args:
            text: Текст, напр. "Встреча завтра в 15:00"
            calendar_id: ID календаря
            
        Returns:
            Созданное событие или None
        """
        try:
            event = self.service.events().quickAdd(
                calendarId=calendar_id,
                text=text
            ).execute()
            return event
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при быстром создании события")

    

    
    def list_calendars(self) -> List[Dict[str, Any]]:
        """
        Получить список всех календарей пользователя
        
        Returns:
            Список календарей
        """
        try:
            calendar_list = self.service.calendarList().list().execute()
            return calendar_list.get('items', [])
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при получении списка календарей")

    
    def get_calendar(self, calendar_id: str = 'primary') -> Optional[Dict[str, Any]]:
        """
        Получить информацию о календаре
        
        Args:
            calendar_id: ID календаря
            
        Returns:
            Данные календаря или None
        """
        try:
            calendar = self.service.calendars().get(
                calendarId=calendar_id
            ).execute()
            return calendar
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при получении календаря")

    
    def create_calendar(self, summary: str, 
                       timezone: str = 'UTC',
                       description: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Создать новый календарь
        
        Args:
            summary: Название календаря
            timezone: Часовой пояс
            description: Описание
            
        Returns:
            Созданный календарь или None
        """
        calendar_body = {
            'summary': summary,
            'timeZone': timezone
        }
        if description:
            calendar_body['description'] = description
        
        try:
            calendar = self.service.calendars().insert(
                body=calendar_body
            ).execute()
            return calendar
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при создании календаря")

    
    def update_calendar(self, calendar_id: str,
                       summary: Optional[str] = None,
                       description: Optional[str] = None,
                       timezone: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Обновить календарь
        
        Args:
            calendar_id: ID календаря
            summary: Новое название
            description: Новое описание
            timezone: Новый часовой пояс
            
        Returns:
            Обновленный календарь или None
        """
        try:
            calendar = self.service.calendars().get(
                calendarId=calendar_id
            ).execute()
            
            if summary:
                calendar['summary'] = summary
            if description:
                calendar['description'] = description
            if timezone:
                calendar['timeZone'] = timezone
            
            updated_calendar = self.service.calendars().update(
                calendarId=calendar_id,
                body=calendar
            ).execute()
            return updated_calendar
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при обновлении календаря")

    
    def delete_calendar(self, calendar_id: str) -> bool:
        """
        Удалить календарь (нельзя удалить primary)
        
        Args:
            calendar_id: ID календаря
            
        Returns:
            True если успешно удалён
        """
        try:
            self.service.calendars().delete(
                calendarId=calendar_id
            ).execute()
            return True
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при удалении календаря")

    

    
    def list_acl(self, calendar_id: str = 'primary') -> List[Dict[str, Any]]:
        """
        Получить список правил доступа к календарю
        
        Args:
            calendar_id: ID календаря
            
        Returns:
            Список правил ACL
        """
        try:
            acl = self.service.acl().list(calendarId=calendar_id).execute()
            return acl.get('items', [])
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при получении ACL списка")

    
    def add_acl_rule(self, user_email: str, 
                    role: str = 'reader',
                    calendar_id: str = 'primary') -> Optional[Dict[str, Any]]:
        """
        Добавить права доступа пользователю
        
        Args:
            user_email: Email пользователя
            role: Роль ('owner', 'writer', 'reader', 'freeBusyReader')
            calendar_id: ID календаря
            
        Returns:
            Созданное правило или None
        """
        rule_body = {
            'scope': {
                'type': 'user',
                'value': user_email
            },
            'role': role
        }
        
        try:
            rule = self.service.acl().insert(
                calendarId=calendar_id,
                body=rule_body
            ).execute()
            return rule
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при добавлении ACL правила")

    
    def delete_acl_rule(self, rule_id: str, 
                       calendar_id: str = 'primary') -> bool:
        """
        Удалить правило доступа
        
        Args:
            rule_id: ID правила (обычно 'user:email@example.com')
            calendar_id: ID календаря
            
        Returns:
            True если успешно удалено
        """
        try:
            self.service.acl().delete(
                calendarId=calendar_id,
                ruleId=rule_id
            ).execute()
            return True
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при удалении ACL правила")

    

    
    def check_freebusy(self, calendars: List[str],
                      time_min: str,
                      time_max: str,
                      timezone: str = 'UTC') -> Optional[Dict[str, Any]]:
        """
        Проверить занятость календарей в указанный период
        
        Args:
            calendars: Список ID календарей для проверки
            time_min: Начало периода (RFC3339)
            time_max: Конец периода
            timezone: Часовой пояс
            
        Returns:
            Информация о занятости или None
        """
        body = {
            'timeMin': time_min,
            'timeMax': time_max,
            'timeZone': timezone,
            'items': [{'id': cal_id} for cal_id in calendars]
        }
        
        try:
            result = self.service.freebusy().query(body=body).execute()
            return result
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при проверке занятости (freebusy)")

    

    
    def get_colors(self) -> Optional[Dict[str, Any]]:
        """
        Получить доступные цвета для событий и календарей
        
        Returns:
            Словарь с цветами или None
        """
        try:
            colors = self.service.colors().get().execute()
            return colors
        except HttpError as error:
            self._handle_http_error(error, "Ошибка при получении доступных цветов")


    def _handle_http_error(self, error: HttpError, message: str) -> None:
        logger.error(f"Google Calendar API error: {message}: {error}")

        status = getattr(error.resp, "status", None)

        if status == 401:
            raise GoogleCalendarException("Токен истёк или невалиден. Требуется повторная авторизация.")
        if status == 403:
            raise GoogleCalendarException("Недостаточно прав для доступа к Google Calendar.")
        if status == 404:
            raise GoogleCalendarException("Запрашиваемый ресурс не найден.")

        raise GoogleCalendarException(f"{message}: {error}")


google_calendar_service = GoogleCalendarClient()

# @app.get("/auth/login")
# def login():
#     auth_url = google_client.web_auth_url(
#         redirect_uri="http://localhost:8000/auth/callback"
#     )
#     return RedirectResponse(auth_url)

# @app.get("/auth/callback")
# def callback(request: Request):
#     full_url = str(request.url)
#     google_client.web_auth_callback(
#         authorization_response=full_url,
#         redirect_uri="http://localhost:8000/auth/callback"
#     )

#     creds_json = google_client.get_credentials_json()
#     user_id = current_user_id()  # твоя логика (через JWT/сессию)
#     user_service.add_google_credentials(current_user_id, creds_json)




# @app.get("/api/calendar/events")
# def get_events():
#     user_id = current_user_id()
#     ccreds_json = await user_service.get_google_credentials(current_user_id)
#     google_client = GoogleCalendarClient()
#     google_client.set_credentials_from_json(creds_json)
#     user_service.add_google_credentials(user_id, google_client.get_credentials_json()) обновляем
#     events = google_client.list_events()

#     return events
