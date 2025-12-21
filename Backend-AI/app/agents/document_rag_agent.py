from __future__ import annotations

from typing import Optional, Annotated
from datetime import datetime
from loguru import logger

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent

from app.rag.rag_service import RAGService

class DocumentRAGTools:    
    def __init__(self, rag_service: RAGService):
        """
        Args:
            rag_service: Сервис для работы с векторной БД и поиском
        """
        self.rag_service = rag_service
    
    @tool
    async def index_uploaded_documents(
        self,
        state: Annotated[dict, InjectedState],
    ) -> str:
        """Проиндексировать загруженные пользователем документы в RAG систему.
        
        Используй когда пользователь загрузил файлы и хочет их сохранить/проанализировать.
        После индексации документы можно искать через search_user_documents.
        
        Args:
            state: Состояние графа (автоматически инжектится)
        
        Returns:
            Результат индексации с количеством обработанных фрагментов
        """
        try:
            user_id = state["user_id"]
            uploaded_files = state.get("uploaded_files", [])
            
            if not uploaded_files:
                return "❌ Нет загруженных файлов для индексации. Попросите пользователя загрузить документы."
            
            indexed = []
            errors = []
            
            for file_info in uploaded_files:
                object_name = file_info.get("object_name") or file_info.get("file_id")
                filename = file_info.get("filename", "unknown")
                file_type = file_info.get("file_type", "unknown")
                
                if not object_name:
                    logger.warning(f"No object_name for file: {filename}")
                    errors.append(f"Не удалось определить путь к файлу {filename}")
                    continue
                
                try:
                    # Определяем тип файла
                    if file_type == "unknown" and "." in filename:
                        file_type = filename.split(".")[-1].lower()
                    
                    # Метаданные для индексации
                    metadata = {
                        "user_id": user_id,
                        "filename": filename,
                        "file_type": file_type,
                        "source": "minio",
                        "indexed_at": datetime.now().isoformat(),
                    }
                    
                    # Индексируем документ
                    chunks_count = await self.rag_service.index_document_from_minio(
                        object_name=object_name,
                        metadata=metadata
                    )
                    
                    indexed.append(f"✅ {filename} ({chunks_count} фрагментов)")
                    logger.info(f"Indexed document: {filename} ({chunks_count} chunks)")
                    
                except Exception as e:
                    logger.error(f"Failed to index document {filename}: {e}")
                    errors.append(f"❌ Ошибка при индексации {filename}: {str(e)}")
            
            # Формируем ответ
            result = f"📚 Проиндексировано документов: {len(indexed)}\n\n"
            result += "\n".join(indexed)
            
            if errors:
                result += f"\n\n⚠️ Ошибки:\n" + "\n".join(errors)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return f"❌ Ошибка индексации документов: {str(e)}"
    
    @tool
    async def index_text_from_message(
        self,
        state: Annotated[dict, InjectedState],
        text: str,
        title: Optional[str] = None,
    ) -> str:
        """Проиндексировать текст из сообщения пользователя в RAG систему.
        
        Используй этот инструмент когда:
        - Пользователь вставил длинный текст и просит его запомнить
        - Пользователь скопировал статью/заметку для сохранения
        - Текст > 200 символов и содержит полезную информацию
        
        Args:
            state: Состояние графа (автоматически инжектится)
            text: Текст для индексации (минимум 100 символов)
            title: Название/описание текста (опционально)
        
        Returns:
            Результат индексации текста
        """
        try:
            user_id = state["user_id"]
            
            text = text.strip()
            if len(text) < 100:
                return "❌ Текст слишком короткий для индексации (минимум 100 символов)"
            
            # Метаданные для индексации
            metadata = {
                "user_id": user_id,
                "source": "text",
                "title": title or "Текст из чата",
                "indexed_at": datetime.now().isoformat(),
            }
            
            # Индексируем текст
            chunks_count = await self.rag_service.index_text(
                text=text,
                metadata=metadata
            )
            
            logger.info(f"Indexed text: {metadata['title']} ({chunks_count} chunks)")
            
            return f"✅ Текст проиндексирован: {chunks_count} фрагментов\nНазвание: {metadata['title']}"
            
        except Exception as e:
            logger.error(f"Failed to index text: {e}")
            return f"❌ Ошибка индексации текста: {str(e)}"
    
    @tool
    async def search_user_documents(
        self,
        state: Annotated[dict, InjectedState],
        query: str,
        max_results: int = 5,
        use_hybrid_search: bool = False,
    ) -> str:
        """Найти и вернуть релевантные документы пользователя.
        
        Используй этот инструмент для ЛЮБОГО поиска в документах пользователя.
        Возвращает содержимое найденных документов, которое ты можешь использовать
        для ответа на вопросы, извлечения информации или анализа.
        
        Args:
            state: Состояние графа (автоматически инжектится)
            query: Поисковый запрос (вопрос пользователя или тема поиска)
            max_results: Количество документов для возврата (по умолчанию 5)
            use_hybrid_search: true для точного поиска, false для семантического
        
        Returns:
            Содержимое найденных документов с указанием источников
        """
        try:
            user_id = state["user_id"]
            
            if not query.strip():
                return "❌ Укажите поисковый запрос"
            
            # Фильтр по user_id (обязательно)
            filter_dict = {"user_id": user_id}
            
            # Поиск в RAG
            documents = self.rag_service.search(
                query=query,
                k=max_results,
                filter=filter_dict,
                use_hybrid=use_hybrid_search
            )
            
            if not documents:
                return f"❌ Документы по запросу '{query}' не найдены."
            
            # Форматируем результаты с полным содержимым
            context_parts = []
            for i, doc in enumerate(documents, 1):
                filename = doc.metadata.get("filename", "unknown")
                page = doc.metadata.get("page")
                file_type = doc.metadata.get("file_type", "")
                
                source = f"{filename}"
                if file_type:
                    source += f" [{file_type.upper()}]"
                if page:
                    source += f", стр. {page}"
                
                context_parts.append(
                    f"[Источник {i}: {source}]\n{doc.page_content}"
                )
            
            logger.info(f"Found {len(documents)} documents for query: {query[:50]}")
            return "\n\n---\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"❌ Ошибка поиска: {str(e)}"



def create_document_rag_agent(
    rag_service: RAGService,
    llm,
    name: str = "document_rag",
):
    """Создать агента для работы с документами и RAG
    
    Args:
        rag_service: Сервис для работы с векторной БД
        llm: Языковая модель
        name: Имя агента (для supervisor handoff)
    
    Returns:
        Compiled ReAct agent
    """
    tools_instance = DocumentRAGTools(rag_service)
    
    tools = [
        tools_instance.index_uploaded_documents,
        tools_instance.index_text_from_message,
        tools_instance.search_user_documents,
    ]
    
    prompt = (
        "Ты - специалист по работе с документами и поиску информации.\n\n"
        "Твои возможности:\n"
        "- Индексация загруженных документов (PDF, DOCX, TXT, CSV, XLSX)\n"
        "- Индексация текста из сообщений\n"
        "- Семантический поиск по документам пользователя\n\n"
        "Рабочий процесс:\n"
        "1. Если пользователь загрузил документы → index_uploaded_documents()\n"
        "2. Если пользователь вставил длинный текст (>200 символов) и просит запомнить → index_text_from_message()\n"
        "3. Если пользователь задаёт вопрос о документах → search_user_documents()\n\n"
        "Правила поиска:\n"
        "- use_hybrid_search=true для точного поиска (названия, даты, термины)\n"
        "- use_hybrid_search=false для семантического поиска (общие вопросы)\n\n"
        "ВАЖНО: Используй ТОЛЬКО информацию из результатов поиска. "
        "Всегда указывай источники (номер источника из результатов)."
    )
    
    agent = create_react_agent(
        model=llm,
        tools=tools,
        name=name,
        prompt=prompt,
    )
    
    logger.info(f"Created DocumentRAGAgent '{name}' with {len(tools)} tools")
    return agent
