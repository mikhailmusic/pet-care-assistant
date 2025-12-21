from __future__ import annotations

from typing import Optional, Annotated
from datetime import datetime, timezone
from loguru import logger
import json

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from app.integrations.ddg_client import DuckDuckGoClient


class WebSearchTools:
    
    def __init__(self, duckduckgo_service: DuckDuckGoClient):
        self.duckduckgo_service = duckduckgo_service
    
    @tool
    async def search_web(
        self,
        state: Annotated[dict, InjectedState],
        query: str,
        max_results: int = 5,
        recent_only: bool = False,
    ) -> str:
        """Поиск в интернете через DuckDuckGo.
        
        Используй когда нужна актуальная информация, факты, новости, исследования.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            query: Поисковый запрос (краткий и точный, 3-7 слов)
                   Примеры: "кошка чихание причины лечение"
                            "ветеринарная клиника Москва рейтинг"
                            "корм Royal Canin цена отзывы"
            max_results: Количество результатов (по умолчанию 5)
            recent_only: Искать только свежую информацию за последний месяц (для новостей)
        
        Returns:
            JSON string с структурированными результатами поиска
        """
        try:
            # Выполняем поиск через DuckDuckGo
            timelimit = "m" if recent_only else None
            
            results = await self.duckduckgo_service.search(
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
    async def fetch_webpage(
        self,
        state: Annotated[dict, InjectedState],
        url: str,
        max_length: int = 15000,
    ) -> str:
        """Получить полное текстовое содержимое веб-страницы.
        
        Используй когда:
        - Пользователь предоставил конкретную ссылку
        - Нужна детальная информация с конкретной страницы
        - После поиска нужно прочитать полную статью
        
        Args:
            state: Состояние графа (автоматически инжектится)
            url: URL страницы для загрузки
            max_length: Максимальная длина текста (по умолчанию 15000 символов)
        
        Returns:
            JSON string с содержимым страницы
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


def create_web_search_agent(
    duckduckgo_service: DuckDuckGoClient,
    llm,
    name: str = "web_search",
):
    """Создать агента для поиска информации в интернете
    
    Args:
        llm: Языковая модель
        name: Имя агента (для supervisor handoff)
    
    Returns:
        Compiled ReAct agent
    """
    tools_instance = WebSearchTools(duckduckgo_service)
    
    tools = [
        tools_instance.search_web,
        tools_instance.fetch_webpage,
    ]
    
    prompt = (
        "Ты - специалист по поиску информации в интернете.\n\n"
        "Твои возможности:\n"
        "- Поиск через DuckDuckGo (актуальная информация, новости, факты)\n"
        "- Загрузка полного текста веб-страниц\n\n"
        "Рабочий процесс:\n"
        "1. Используй search_web для поиска релевантных страниц\n"
        "2. Выбери 2-3 САМЫХ релевантных результата\n"
        "3. Используй fetch_webpage для загрузки полного текста этих страниц\n"
        "4. Верни структурированные данные (поисковые результаты + загруженные страницы)\n\n"
        "Формируй короткие и точные поисковые запросы (3-7 слов).\n"
        "Для русскоязычных запросов используй русский язык."
    )
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        name=name,
        prompt=prompt,
    )
    
    logger.info(f"Created WebSearchAgent '{name}' with {len(tools)} tools")
    return agent