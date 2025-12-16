from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
from contextvars import ContextVar

from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.rag.rag_service import RAGService, get_rag_service
from app.integrations.gigachat_client import GigaChatClient
from app.config import settings


@dataclass
class DocumentRAGContext:
    """Контекст для инструментов работы с документами"""
    user_id: int
    current_pet_id: Optional[int] = None
    current_pet_name: str = ""
    uploaded_files: List[Dict[str, Any]] = field(default_factory=list)


_document_rag_context: ContextVar[Optional[DocumentRAGContext]] = ContextVar(
    '_document_rag_context', 
    default=None
)

_rag_service: ContextVar[Optional[RAGService]] = ContextVar('_rag_service', default=None)


def _get_context() -> DocumentRAGContext:
    """Get the current document RAG context from ContextVar"""
    ctx = _document_rag_context.get()
    if ctx is None:
        raise RuntimeError("DocumentRAG context not set. This should not happen.")
    return ctx


def _get_rag_service() -> RAGService:
    """Get RAG service from ContextVar"""
    service = _rag_service.get()
    if service is None:
        raise RuntimeError("RAG service not set. This should not happen.")
    return service


# ============================================================================
# TOOLS
# ============================================================================

@tool
async def index_uploaded_documents() -> str:
    """Проиндексировать загруженные пользователем документы в RAG систему.
    
    Используй когда пользователь загрузил файлы и хочет их сохранить/проанализировать.
    После индексации документы можно искать через search_user_documents.
    
    Returns:
        Результат индексации с количеством обработанных фрагментов
    """
    try:
        ctx = _get_context()
        rag_service = _get_rag_service()
        
        if not ctx.uploaded_files:
            return "❌ Нет загруженных файлов для индексации. Попросите пользователя загрузить документы."
        
        indexed = []
        errors = []
        
        for file_info in ctx.uploaded_files:
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
                    "user_id": ctx.user_id,
                    "filename": filename,
                    "file_type": file_type,
                    "source": "minio",
                    "indexed_at": datetime.now().isoformat(),
                }
                
                if ctx.current_pet_id:
                    metadata["pet_id"] = ctx.current_pet_id
                    metadata["pet_name"] = ctx.current_pet_name
                
                # Индексируем документ
                chunks_count = await rag_service.index_document_from_minio(
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
    text: str,
    title: Optional[str] = None,
) -> str:
    """Проиндексировать текст из сообщения пользователя в RAG систему.
    
    Используй этот инструмент когда:
    - Пользователь вставил длинный текст и просит его запомнить
    - Пользователь скопировал статью/заметку для сохранения
    - Текст > 200 символов и содержит полезную информацию
    
    Args:
        text: Текст для индексации (минимум 100 символов)
        title: Название/описание текста (опционально)
    
    Returns:
        Результат индексации текста
    """
    try:
        ctx = _get_context()
        rag_service = _get_rag_service()
        
        text = text.strip()
        if len(text) < 100:
            return "❌ Текст слишком короткий для индексации (минимум 100 символов)"
        
        # Метаданные для индексации
        metadata = {
            "user_id": ctx.user_id,
            "source": "text",
            "title": title or "Текст из чата",
            "indexed_at": datetime.now().isoformat(),
        }
        
        if ctx.current_pet_id:
            metadata["pet_id"] = ctx.current_pet_id
            metadata["pet_name"] = ctx.current_pet_name
        
        # Индексируем текст
        chunks_count = await rag_service.index_text(
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
    query: str,
    max_results: int = 5,
    use_hybrid_search: bool = False,
) -> str:
    """Найти и вернуть релевантные документы пользователя.
    
    Используй этот инструмент для ЛЮБОГО поиска в документах пользователя.
    Возвращает содержимое найденных документов, которое ты можешь использовать
    для ответа на вопросы, извлечения информации или анализа.
    
    Args:
        query: Поисковый запрос (вопрос пользователя или тема поиска)
        max_results: Количество документов для возврата (по умолчанию 5)
        use_hybrid_search: true для точного поиска, false для семантического
    
    Returns:
        Содержимое найденных документов с указанием источников
    """
    try:
        ctx = _get_context()
        rag_service = _get_rag_service()
        
        if not query.strip():
            return "❌ Укажите поисковый запрос"
        
        # Фильтр по user_id (обязательно)
        filter_dict = {"user_id": ctx.user_id}
        
        if ctx.current_pet_id:
            filter_dict["pet_id"] = ctx.current_pet_id
        
        # Поиск в RAG
        documents = rag_service.search(
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


class DocumentRAGAgent:
    """Агент для работы с документами и RAG через LangChain tools"""
    
    def __init__(
        self,
        llm=None,
        use_hybrid_retriever: bool = False
    ):
        """
        Args:
            llm: LLM для агента
            use_hybrid_retriever: Использовать гибридный retriever по умолчанию
        """
        self.llm = llm or GigaChatClient().llm
        
        # Инициализируем RAG сервис
        self.rag_service = get_rag_service(use_hybrid_retriever=use_hybrid_retriever)
        
        # Список инструментов
        self.tools = [
            index_uploaded_documents,
            index_text_from_message,
            search_user_documents,
        ]
        
        logger.info(f"DocumentRAGAgent initialized (hybrid={use_hybrid_retriever})")
    
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
            context: Контекст (uploaded_files, current_pet_id, current_pet_name)
        
        Returns:
            Ответ агента
        """
        context = context or {}
        token = None
        rag_token = None
        
        try:
            # Создаём контекст для tools (только данные пользователя)
            tool_context = DocumentRAGContext(
                user_id=user_id,
                current_pet_id=context.get("current_pet_id"),
                current_pet_name=context.get("current_pet_name", ""),
                uploaded_files=context.get("uploaded_files", [])
            )
            
            # Устанавливаем контексты
            token = _document_rag_context.set(tool_context)
            rag_token = _rag_service.set(self.rag_service)
            
            # Формируем system prompt
            files_info = ""
            if tool_context.uploaded_files:
                files_list = [
                    f"- {f.get('filename', 'unknown')}" 
                    for f in tool_context.uploaded_files[:5]
                ]
                files_info = "Загруженные файлы:\n" + "\n".join(files_list)
                if len(tool_context.uploaded_files) > 5:
                    files_info += f"\n... и ещё {len(tool_context.uploaded_files) - 5}"
            
            system_prompt = f"""Ты - помощник по работе с документами и медицинскими записями для владельцев домашних животных.

Контекст:
- Пользователь ID: {user_id}
- Питомец: {tool_context.current_pet_name or "не указан"} (ID: {tool_context.current_pet_id or "не указан"})
{files_info if files_info else "- Нет загруженных файлов в текущем сообщении"}

**Доступные инструменты:**

1. **index_uploaded_documents** - Индексировать загруженные пользователем файлы
   Используй когда пользователь загрузил документы

2. **index_text_from_message** - Сохранить текст из сообщения
   Используй когда пользователь вставил длинный текст (>200 символов) и просит запомнить

3. **search_user_documents** - Найти документы по запросу
   Используй для ЛЮБОГО поиска информации в документах пользователя
   Возвращает содержимое найденных документов - используй его для ответов!

**Как работать с документами:**

1. Если пользователь загрузил файлы → index_uploaded_documents()

2. Если пользователь задаёт вопрос о документах:
   - Вызови search_user_documents(query="вопрос пользователя", max_results=5)
   - Проанализируй возвращённый контекст
   - Ответь на вопрос используя ТОЛЬКО информацию из контекста
   - Укажи источники (номера источников из результатов)

3. Если нужно найти конкретные данные (показатели, лекарства, даты):
   - Вызови search_user_documents(query="что искать", max_results=10, use_hybrid_search=true)
   - Извлеки данные из результатов
   - Структурируй ответ (списки, таблицы)

**Правила use_hybrid_search:**
- use_hybrid_search=true → для точного поиска по ключевым словам, названиям, датам
  Примеры: "прививка от бешенства", "анализ от 15 декабря", "препарат Римадил"
- use_hybrid_search=false → для семантического поиска (по смыслу)
  Примеры: "что-то про аллергию", "документы о здоровье", "рекомендации ветеринара"

**Примеры работы:**

Пользователь: "Что написано в анализе крови?"
1. search_user_documents(query="анализ крови", max_results=3)
2. Анализируешь результаты
3. Отвечаешь: "Согласно Источнику 1 (анализ_крови.pdf)..."

Пользователь: "Выпиши все назначенные лекарства"
1. search_user_documents(query="назначенные лекарства препараты", max_results=10, use_hybrid_search=true)
2. Извлекаешь из результатов все упоминания лекарств
3. Структурируешь: "**Назначенные лекарства:**\n1. Римадил...\n2. ..."

Пользователь: "Какие рекомендации дал ветеринар?"
1. search_user_documents(query="рекомендации ветеринар", max_results=5)
2. Находишь рекомендации в результатах
3. Отвечаешь с указанием источника

**Важно:**
- ВСЕГДА используй только информацию из результатов search_user_documents
- ВСЕГДА указывай источники (номер источника из результатов)
- Если информации нет в результатах - честно скажи об этом
- Не выдумывай информацию

Отвечай дружелюбно и структурированно!"""
            
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
            return result.get("output", "Обработан запрос о документах")
            
        except Exception as e:
            logger.exception(f"DocumentRAGAgent error for user {user_id}")
            return f"❌ Ошибка при работе с документами: {str(e)}"
        finally:
            if token is not None:
                _document_rag_context.reset(token)
            if rag_token is not None:
                _rag_service.reset(rag_token)