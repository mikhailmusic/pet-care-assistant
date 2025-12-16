import string
from typing import List, Dict
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_chroma import Chroma
from loguru import logger


class RetrieverFactory:
    
    @staticmethod
    def create_embedding_retriever(vector_store: Chroma,k: int = 5, filter: Dict | None = None):
        search_kwargs = {"k": k}
        
        if filter:
            search_kwargs["filter"] = filter
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
        logger.info(f"Embedding retriever created with k={k}, filter={filter}")
        return retriever
    
    @staticmethod
    def create_bm25_retriever(documents: List[Document],k: int = 5,filter: Dict | None = None):
        
        if filter:
            filtered_docs = [
                doc for doc in documents
                if all(doc.metadata.get(key) == value for key, value in filter.items())
            ]
            logger.info(
                f"BM25 retriever: filtered {len(documents)} -> {len(filtered_docs)} docs "
                f"by {filter}"
            )
        else:
            filtered_docs = documents
        
        if not filtered_docs:
            logger.warning("BM25 retriever created with 0 documents after filtering!")
        
        bm25_retriever = BM25Retriever.from_documents(
            documents=filtered_docs,
            preprocess_func=RetrieverFactory.tokenize,
            k=k,
        )
        
        logger.info(f"BM25 retriever created with k={k}, docs={len(filtered_docs)}")
        return bm25_retriever
    
    
    @staticmethod
    def create_hybrid_retriever(
        vector_store: Chroma,
        documents: List[Document],
        embedding_k: int = 3,
        bm25_k: int = 2,
        embedding_weight: float = 0.6,
        bm25_weight: float = 0.4,
        filter: Dict | None = None
    ):
        
        embedding_retriever = RetrieverFactory.create_embedding_retriever(
            vector_store=vector_store,
            k=embedding_k,
            filter=filter
        )
        
        bm25_retriever = RetrieverFactory.create_bm25_retriever(
            documents=documents,
            k=bm25_k,
            filter=filter
        )
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[embedding_retriever, bm25_retriever],
            weights=[embedding_weight, bm25_weight],
        )
        
        logger.info(
            f"Hybrid retriever created: embedding_k={embedding_k}, bm25_k={bm25_k}, "
            f"weights=[{embedding_weight}, {bm25_weight}], filter={filter}"
        )
        
        return ensemble_retriever
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Токенизация текста для BM25 retriever"""
        return text.lower().translate(str.maketrans("", "", string.punctuation)).split()    