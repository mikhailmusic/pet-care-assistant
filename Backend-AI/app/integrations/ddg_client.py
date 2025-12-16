from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio

from loguru import logger

from ddgs import DDGS
from app.config import settings
from app.utils.exceptions import ExternalServiceException


class DuckDuckGoClient:
    def __init__(
        self,
        region: Optional[str] = None,
        safesearch: Optional[str] = None,
        timelimit: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> None:
        self.region = region or settings.DDG_REGION
        self.safesearch = safesearch or settings.DDG_SAFESEARCH
        self.timelimit = timelimit or settings.DDG_TIMELIMIT
        self.max_results = max_results or settings.DDG_MAX_RESULTS

        logger.info(
            "DuckDuckGoClient initialized: region=%s, safesearch=%s, timelimit=%s, max_results=%s",
            self.region,self.safesearch,self.timelimit,self.max_results)

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: Optional[str] = None,
        safesearch: Optional[str] = None,
        timelimit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список словарей вида:
        {
          "title": str,
          "href": str,
          "body": str,
        }
        """

        _max_results = max_results or self.max_results
        _region = region or self.region
        _safesearch = safesearch or self.safesearch
        _timelimit = timelimit or self.timelimit

        logger.info(
            "DuckDuckGo search: q='%s', max_results=%s, region=%s, safesearch=%s, timelimit=%s",
            query,_max_results, _region,_safesearch,_timelimit,)

        def _sync_search() -> List[Dict[str, Any]]:
            try:
                with DDGS() as ddgs:
                    results = list(
                        ddgs.text(query,region=_region,safesearch=_safesearch,timelimit=_timelimit,max_results=_max_results,)
                    )
                return results
            except Exception as e:
                logger.error(f"DuckDuckGo search error: {e}")
                raise ExternalServiceException("DuckDuckGo", str(e)) from e

        return await asyncio.to_thread(_sync_search)


duckduckgo_service = DuckDuckGoClient()