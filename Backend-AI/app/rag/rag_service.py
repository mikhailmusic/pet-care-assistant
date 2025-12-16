from typing import List, Dict
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.chains.retrieval import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from app.config import settings
from .embeddings import embeddings_service
from .document_processor import document_processor
from .retrievers import RetrieverFactory
from app.integrations import minio_service
from app.integrations.gigachat_client import GigaChatClient



RAG_PROMPT_TEMPLATE = """Ты — точный и сдержанный ассистент.

Используй следующий контекст для ответа на вопрос. Если информации недостаточно, честно скажи об этом.

Контекст:
{context}

Вопрос: {input}

Ответ:"""



class RAGService:
    def __init__(self, collection_name: str | None = None, use_hybrid_retriever: bool = False):

        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.use_hybrid_retriever = use_hybrid_retriever
        
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings_service.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )
        
        self.rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        logger.info(
            f"RAGService initialized: collection='{self.collection_name}', "
            f"persist_dir='{settings.CHROMA_PERSIST_DIRECTORY}', "
            f"hybrid={use_hybrid_retriever}"
        )
    
    async def index_document_from_minio(self,object_name: str,metadata: Dict | None = None) -> int:

        try:
            file_metadata = await minio_service.get_file_metadata(object_name)
            original_filename = file_metadata.get("original_filename") or object_name.split("/")[-1]
            
            file_type = Path(original_filename).suffix[1:].lower()
            if not file_type:
                file_type = "unknown"
    
            file_data = await minio_service.download_file(object_name)
            file_data.seek(0)
            file_bytes = file_data.read()
            
            
            documents = await document_processor.parse_from_bytes(
                file_bytes=file_bytes,
                filename=original_filename
            )
            
            split_docs = document_processor.split_documents(documents)
            
            base_metadata = {
                "filename": original_filename,
                "file_type": file_type,
                "object_name": object_name,
                "source": "minio",
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            for doc in split_docs:
                doc.metadata.update(base_metadata)
            
            ids = [str(uuid.uuid4()) for _ in split_docs]
            
            self.vector_store.add_documents(
                documents=split_docs,
                ids=ids
            )
            
            logger.info(
                f"Document indexed from MinIO: {object_name} -> {len(split_docs)} chunks"
            )
            
            return len(split_docs)
            
        except Exception as e:
            logger.error(f"Failed to index document from MinIO {object_name}: {e}")
            raise
    
    async def index_text(self,text: str, metadata: Dict | None = None) -> int:

        try:
            split_docs = document_processor.split_text(text)
            
            if not split_docs:
                logger.warning("No chunks created from text")
                return 0

            base_metadata = {
                "source": "text",
                "file_type": "text",
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            for doc in split_docs:
                doc.metadata.update(base_metadata)
            
            ids = [str(uuid.uuid4()) for _ in split_docs]
            
            self.vector_store.add_documents(
                documents=split_docs,
                ids=ids
            )
            
            logger.info(f"Text indexed: {len(split_docs)} chunks")
            
            return len(split_docs)
            
        except Exception as e:
            logger.error(f"Failed to index text: {e}")
            raise
    
    def search(self,query: str, k: int | None = None,filter: Dict | None = None, use_hybrid: bool = False) -> List[Document]:
        k = k or settings.TOP_K_RESULTS
        
        try:
            if use_hybrid:
                user_docs = self._get_documents_for_bm25(filter=filter)
                
                if not user_docs:
                    logger.warning(f"No documents for hybrid retriever with filter={filter}, falling back to embedding")
                    retriever = RetrieverFactory.create_embedding_retriever(vector_store=self.vector_store,k=k,filter=filter)
                else:
                    retriever = RetrieverFactory.create_hybrid_retriever(vector_store=self.vector_store,documents=user_docs,filter=filter)
                
                # invoke / get_relevant_documents
                results = retriever.get_relevant_documents(query)
            else:
                # Простой эмбеддинг поиск
                results = self.vector_store.similarity_search(query=query,k=k,filter=filter)
            
            logger.info(
                f"Search: query='{query[:50]}...', "
                f"results={len(results)}, hybrid={use_hybrid}, filter={filter}"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def query_with_llm(self, query: str, llm, k: int | None = None, filter: Dict | None = None) -> Dict:

        k = k or settings.TOP_K_RESULTS
        
        try:
            
            if self.use_hybrid_retriever:
                user_docs = self._get_documents_for_bm25(filter=filter)
                
                if not user_docs:
                    logger.warning(f"No documents for hybrid retriever with filter={filter}, falling back to embedding")
                    retriever = RetrieverFactory.create_embedding_retriever(vector_store=self.vector_store, k=k, filter=filter)
                else:
                    retriever = RetrieverFactory.create_hybrid_retriever(vector_store=self.vector_store, documents=user_docs, filter=filter)
            else:
                retriever = RetrieverFactory.create_embedding_retriever(vector_store=self.vector_store, k=k, filter=filter)
            
            if isinstance(llm, GigaChatClient):
                document_chain = create_stuff_documents_chain(
                    llm=llm.llm,
                    prompt=self.rag_prompt
                )
            else:
                document_chain = create_stuff_documents_chain(
                    llm=llm,
                    prompt=self.rag_prompt
                )
            
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            result = retrieval_chain.invoke({"input": query})
            
            response = {
                "answer": result["answer"],
                "context": result.get("context", []),
                "metadata": {
                    "query": query,
                    "sources_count": len(result.get("context", [])),
                    "retriever_type": "hybrid" if self.use_hybrid_retriever else "embedding",
                    "filter": filter
                }
            }
            
            logger.info(
                f"RAG query completed: query='{query[:50]}...', "
                f"sources={len(result.get('context', []))}, filter={filter}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            raise
    
    def _get_documents_for_bm25(self, filter: Dict | None = None, limit: int = 1000) -> List[Document]:
        try:
            collection = self.vector_store._collection
            
            results = collection.get(
                where=filter,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            documents = []
            for i, doc_text in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                documents.append(Document(page_content=doc_text, metadata=metadata))
            
            logger.info(
                f"Retrieved {len(documents)} documents for BM25 with filter={filter}"
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get documents for BM25: {e}")
            return []



    def delete_documents(self, ids: List[str]) -> bool:
        """Удаление документов по ID"""
        try:
            self.vector_store.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def delete_by_filter(self, filter: Dict) -> bool:
        """Удаление документов по фильтру"""
        try:
            self.vector_store.delete(where=filter)
            logger.info(f"Deleted documents by filter: {filter}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete by filter: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Статистика коллекции"""
        collection = self.vector_store._collection
        
        return {
            "collection_name": self.collection_name,
            "total_documents": collection.count(),
            "use_hybrid_retriever": self.use_hybrid_retriever,
            "persist_directory": settings.CHROMA_PERSIST_DIRECTORY,
        }


get_rag_service = RAGService()


