from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from app.config import settings


class EmbeddingsService:
    def __init__(self, model_name: str | None = None, device: str | None = None, normalize_embeddings: bool = False):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE

        self.normalize_embeddings = normalize_embeddings

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")

        model_kwargs = {"device": self.device}
        encode_kwargs = {"normalize_embeddings": normalize_embeddings}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        logger.info(f"Embedding model loaded: {self.model_name}")
    

embeddings_service = EmbeddingsService()