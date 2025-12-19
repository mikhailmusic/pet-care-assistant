from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from loguru import logger
from contextvars import ContextVar
import json
from datetime import datetime, timezone


from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.integrations.gigachat_client import GigaChatClient
from app.integrations import duckduckgo_service
from app.config import settings


@dataclass
class WebSearchContext:
    """Контекст для Web Search Agent"""
    user_id: int
    current_pet_name: str = ""
    current_pet_species: str = ""


_web_search_context: ContextVar[Optional[WebSearchContext]] = ContextVar(
    '_web_search_context',
    default=None
)


def _get_context() -> WebSearchContext:
    """Get the current web search context from ContextVar"""
    ctx = _web_search_context.get()
    if ctx is None:
        raise RuntimeError("WebSearch context not set. This should not happen.")
    return ctx


@tool
async def search_web(
    query: str,
    max_results: int = 5,
    recent_only: bool = False,
) -> str:
    """Поиск в интернете через DuckDuckGo.
    
    Используй когда нужна актуальная информация, факты, новости, исследования.
    Агент САМ формирует оптимальный поисковый запрос.
    
    Args:
        query: Поисковый запрос (краткий и точный, 3-7 слов)
               Примеры: "кошка чихание причины лечение"
                        "ветеринарная клиника Москва рейтинг"
                        "корм Royal Canin цена отзывы"
        max_results: Количество результатов (по умолчанию 5)
        recent_only: Искать только свежую информацию за последний месяц (для новостей)
    
    Returns:
        JSON string с структурированными результатами:
        {
          "query": str,
          "provider": "duckduckgo",
          "retrieved_at": ISO8601 timestamp,
          "results": [
              {
                "rank": int,
                "title": str,
                "url": str,
                "snippet": str
              }
          ]
        }
    """
    try:
        # Выполняем поиск через DuckDuckGo
        timelimit = "m" if recent_only else None
        
        results = await duckduckgo_service.search(
            query=query,
            max_results=max_results,
            timelimit=timelimit,
            region="ru-ru"
        )
        
        # Формируем структурированный результат
        structured_result = {
            "query": query,
            "provider": "duckduckgo",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "results": []
        }
        
        for i, result in enumerate(results, 1):
            structured_result["results"].append({
                "rank": i,
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")
            })
        
        logger.info(
            f"Web search: query='{query}', recent_only={recent_only}, "
            f"found={len(results)} results"
        )
        
        return json.dumps(structured_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        error_result = {
            "query": query,
            "provider": "duckduckgo",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "results": []
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


@tool
async def fetch_webpage(url: str, max_length: int = 15000) -> str:
    """Получить полное текстовое содержимое веб-страницы.

    Используй когда:
    - Пользователь предоставил конкретную ссылку
    - Нужна детальная информация с конкретной страницы
    - После поиска нужно прочитать полную статью

    Args:
        url: URL страницы для загрузки
        max_length: Максимальная длина текста (по умолчанию 15000 символов)

    Returns:
        JSON string с содержимым страницы:
        {
          "url": str,
          "title": str|null,
          "retrieved_at": ISO8601 timestamp,
          "content": str,
          "truncated": bool,
          "content_length": int
        }
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Загружаем страницу
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Парсим HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Извлекаем title
        title = None
        if soup.title:
            title = soup.title.string
        
        # Удаляем ненужные элементы
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
        
        # Извлекаем текст
        text = soup.get_text()
        
        # Очищаем от лишних пробелов
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Проверяем нужно ли обрезать
        original_length = len(text)
        truncated = original_length > max_length
        
        if truncated:
            text = text[:max_length]
        
        # Формируем структурированный результат
        structured_result = {
            "url": url,
            "title": title,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "content": text,
            "truncated": truncated,
            "content_length": original_length
        }
        
        logger.info(
            f"Fetched webpage: url={url}, length={original_length}, "
            f"truncated={truncated}"
        )
        
        return json.dumps(structured_result, ensure_ascii=False, indent=2)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        error_result = {
            "url": url,
            "title": None,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "error": f"Network error: {str(e)}",
            "content": "",
            "truncated": False,
            "content_length": 0
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Error parsing {url}: {e}")
        error_result = {
            "url": url,
            "title": None,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "error": f"Parse error: {str(e)}",
            "content": "",
            "truncated": False,
            "content_length": 0
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


# ============================================================================
# WEB SEARCH AGENT
# ============================================================================

class WebSearchAgent:
    """Агент для поиска информации в интернете через DuckDuckGo
    
    Возвращает структурированные результаты для оркестратора.
    Оркестратор решает что делать с результатами:
    - Показать пользователю
    - Передать в DocumentRAGAgent для индексации
    - Использовать как контекст для ответа
    """
    
    def __init__(self, llm=None):
        """
        Args:
            llm: LLM для агента
        """
        self.llm = llm or GigaChatClient().llm
        
        # Список инструментов
        self.tools = [
            search_web,
            fetch_webpage
        ]
        
        logger.info("WebSearchAgent initialized with 2 tools (DuckDuckGo)")
    
    async def process(
        self,
        user_id: int,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Обработать запрос пользователя
        
        Args:
            user_id: ID пользователя
            user_message: Сообщение пользователя
            context: Контекст (current_pet_name, current_pet_species)
        
        Returns:
            Результаты поиска (НЕ проиндексированные, только текст)
        """
        context = context or {}
        token = None
        
        try:
            # Создаём контекст
            tool_context = WebSearchContext(
                user_id=user_id,
                current_pet_name=context.get("current_pet_name", ""),
                current_pet_species=context.get("current_pet_species", ""),
            )
            
            # Устанавливаем контекст
            token = _web_search_context.set(tool_context)
            
            # Информация о питомце для контекста
            pet_info = ""
            if tool_context.current_pet_name:
                pet_info = f"\n🐾 Текущий питомец: {tool_context.current_pet_name}"
                if tool_context.current_pet_species:
                    pet_info += f" ({tool_context.current_pet_species})"
            
            # System prompt
            system_prompt = f"""Ты - эксперт по поиску и сбору информации в интернете для владельцев домашних животных.

Пользователь ID: {user_id}{pet_info}

**Твои инструменты (всего 2):**

1. **search_web(query, max_results, recent_only)** - Поиск в DuckDuckGo
   - Ты САМ формируешь оптимальный поисковый запрос
   - Делай запросы точными и краткими (3-7 слов)
   - Используй русский для русскоязычных запросов
   - Для питомцев добавляй вид в запрос

2. **fetch_webpage(url, max_length=15000)** - Загрузка полного текста страницы
   - Загружает ВЕСЬ текстовый контент со страницы (до 15000 символов)
   - Используй для 2-3 САМЫХ РЕЛЕВАНТНЫХ результатов поиска
   - Возвращает полный текст со страницы

**КРИТИЧЕСКИ ВАЖНО - Обязательный рабочий процесс:**

Шаг 1: Выполни search_web для поиска релевантных страниц
Шаг 2: Выбери 2-3 САМЫХ РЕЛЕВАНТНЫХ результата и вызови fetch_webpage для каждого из них
Шаг 3: Верни структурированные данные (БЕЗ анализа - это сделает оркестратор)

**Формат итогового ответа:**
КРИТИЧЕСКИ ВАЖНО: Возвращай ВАЛИДНЫЙ JSON без форматирования. Используй \\n для переводов строк внутри текста.

Структура JSON:
{{{{
  "search_results": [
    {{{{
      "rank": 1,
      "title": "Название страницы",
      "url": "https://...",
      "snippet": "Краткое описание из поиска"
    }}}}
  ],
  "loaded_pages": [
    {{{{
      "url": "https://...",
      "title": "Название страницы",
      "content": "Полный текст страницы (загруженный через fetch_webpage)"
    }}}}
  ],
  "summary": "Краткое резюме в 1-2 предложениях: что нашел и какие страницы загрузил"
}}}}

**ВАЖНО:**
- В "search_results" включай ВСЕ результаты поиска (rank, title, url, snippet)
- В "loaded_pages" включай ПОЛНЫЙ текст только для 2-3 самых релевантных страниц (те, которые ты загрузил через fetch_webpage)
- В "summary" напиши краткое резюме: что искал, сколько нашел, какие страницы загрузил
- НЕ делай анализ контента - это сделает оркестратор
- Твоя задача: НАЙТИ и ЗАГРУЗИТЬ релевантную информацию

**Правила формирования запросов:**

  **Хорошие запросы:**
- "кошка чихание причины лечение ветеринар"
- "собака понос что делать"
- "ветеринарная клиника Москва рейтинг"
- "корм Royal Canin состав отзывы"
- "прививка от бешенства кошка цена"

  **Плохие запросы:**
- "что делать если кошка чихает" (слишком длинно)
- "cat sneezing" (для русскоязычных - используй русский)
- "чихание" (слишком общо, добавь контекст)

**Специализация запросов:**

1. **Вопросы о здоровье питомцев:**
   - Добавляй: вид животного + симптом + "ветеринар"
   - Пример: "собака рвота понос ветеринар лечение"

2. **Поиск клиник:**
   - Добавляй: "ветеринарная клиника" + город + "отзывы рейтинг"
   - Пример: "ветеринарная клиника Санкт-Петербург отзывы"

3. **Новости (recent_only=True):**
   - Добавляй: тема + "новости"
   - Пример: "корм для кошек новости отзыв"

4. **Цены и покупки:**
   - Добавляй: товар + "цена отзывы где купить"
   - Пример: "Royal Canin для кошек цена отзывы"

**Когда НЕ использовать поиск:**
- Общие вопросы (можешь ответить сам)
- Личная информация пользователя
- Вопросы о питомцах пользователя (есть другие агенты)

**Когда ИСПОЛЬЗОВАТЬ поиск:**
- Актуальная информация (новости, события)
- Медицинская информация
- Цены, товары, услуги
- Исследования, факты, статистика
- Пользователь явно просит поискать

**Пример хорошего ответа:**
{{{{
  "search_results": [
    {{{{"rank": 1, "title": "Уход за котом", "url": "https://example.com/1", "snippet": "10 советов..."}}}},
    {{{{"rank": 2, "title": "Средства для котов", "url": "https://example.com/2", "snippet": "Популярные средства..."}}}}
  ],
  "loaded_pages": [
    {{{{"url": "https://example.com/1", "title": "Уход за котом", "content": "Полный текст статьи про уход за котом..."}}}},
    {{{{"url": "https://example.com/2", "title": "Средства для котов", "content": "Полный текст статьи про средства..."}}}}
  ],
  "summary": "Найдено 5 результатов по запросу 'средства ухода за котом'. Загружены 2 наиболее релевантные статьи."
}}}}

**Итоговый результат:**
- Оркестратор получит структурированные данные и сформирует финальный ответ пользователю
- Полный текст страниц может быть использован для индексации в RAG или для формирования ответа"""
            
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
                max_iterations=10,  # Увеличено для загрузки нескольких страниц после поиска
            )
            
            result = await agent_executor.ainvoke({"input": user_message})
            return result.get("output", '{"error": "No output from agent"}')
            
        except Exception as e:
            logger.exception(f"WebSearchAgent error for user {user_id}")
            error_result = {
                "error": str(e),
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            return json.dumps(error_result, ensure_ascii=False)
        finally:
            if token is not None:
                _web_search_context.reset(token)