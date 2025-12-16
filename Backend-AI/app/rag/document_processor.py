import string
from typing import List, Optional
from pathlib import Path
from tempfile import NamedTemporaryFile


from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader, CSVLoader, UnstructuredExcelLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger

from app.config import settings


class DocumentProcessor:    
    SUPPORTED_FORMATS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"}
    
    def __init__(self,chunk_size: Optional[int] = None,chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        logger.info(
            f"DocumentProcessor initialized: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    async def parse_from_bytes(self, file_bytes: bytes, filename: str = "document.txt") -> List[Document]:
        suffix = (Path(filename).suffix or ".txt").lower()
        
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        logger.info(f"Parsing document from bytes: {filename} (type={suffix})")
        
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = Path(tmp.name)
        
        try:
            if suffix == ".pdf":
                loader = UnstructuredPDFLoader(str(tmp_path))
            elif suffix == ".docx":
                loader = UnstructuredWordDocumentLoader(str(tmp_path))
            elif suffix == ".txt":
                loader = TextLoader(str(tmp_path), encoding="utf-8")
            elif suffix == ".csv":
                loader = CSVLoader(str(tmp_path), encoding="utf-8")
            elif suffix in {".xlsx", ".xls"}:
                loader = UnstructuredExcelLoader(str(tmp_path), mode="elements")
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
            
            docs = loader.load()
            
            logger.info(
                f"Document parsed: {filename} -> {len(docs)} sections, "
                f"total chars: {sum(len(d.page_content) for d in docs)}"
            )
            
            return docs
            
        except Exception as e:
            logger.error(f"Failed to parse document {filename}: {e}")
            raise
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove temp file {tmp_path}: {cleanup_err}")


    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.warning("split_documents called with empty list")
            return []
        
        split_docs = self.text_splitter.split_documents(documents)
        
        logger.info(
            f"Documents split: {len(documents)} docs -> {len(split_docs)} chunks "
            f"(avg {len(split_docs) / len(documents):.1f} chunks/doc)"
        )
        
        return split_docs
    
    def split_text(self, text: str) -> List[Document]:

        if not text.strip():
            logger.warning("split_text called with empty text")
            return []
        
        chunks = self.text_splitter.split_text(text)
        
        documents = [ Document(page_content=chunk) for chunk in chunks]
        
        logger.info(
            f"Text split: {len(text)} chars -> {len(documents)} chunks "
            f"(avg {len(text) / len(documents):.0f} chars/chunk)"
        )
        
        return documents


document_processor = DocumentProcessor()