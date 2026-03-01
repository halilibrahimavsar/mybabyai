import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

import sys


from src.utils.config import Config
from src.utils.logger import get_logger


class MemoryManager:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_logger("memory")

        self.db_path = str(self.config.get_path("memory.vector_db_path", "data/chroma"))
        self.embedding_model_name = self.config.get(
            "memory.embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.chunk_size = self.config.get("memory.chunk_size", 512)
        self.chunk_overlap = self.config.get("memory.chunk_overlap", 50)
        self.max_history = self.config.get("memory.max_history", 100)

        self._setup_embedding_model()
        self._setup_chroma()

        self.conversation_history: List[Dict[str, str]] = []

    def _setup_embedding_model(self) -> None:
        self.logger.info(f"Embedding modeli yükleniyor: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name
            )
        )

    def _setup_chroma(self) -> None:
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=self.db_path, settings=Settings(anonymized_telemetry=False)
        )

        self.documents_collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        self.conversations_collection = self.chroma_client.get_or_create_collection(
            name="conversations",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        self.logger.info("ChromaDB başarıyla kuruldu")

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        chunks = self._chunk_text(text)

        ids = []
        metadatas = []
        documents = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id or 'doc'}_chunk_{i}" if doc_id else None
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = i

            ids.append(chunk_id or f"doc_{len(documents)}_{i}")
            metadatas.append(chunk_metadata)
            documents.append(chunk)

        self.documents_collection.add(ids=ids, documents=documents, metadatas=metadatas)

        self.logger.info(f"Belge eklendi: {len(chunks)} chunk")
        return ids[0] if ids else ""

    def add_documents_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        all_ids = []
        all_documents = []
        all_metadatas = []

        for i, text in enumerate(texts):
            chunks = self._chunk_text(text)
            for j, chunk in enumerate(chunks):
                chunk_id = (
                    f"{ids[i] if ids else 'doc'}_{i}_chunk_{j}"
                    if ids
                    else f"batch_doc_{i}_{j}"
                )
                chunk_metadata = (metadatas[i].copy() if metadatas else {}) | {
                    "source_index": i,
                    "chunk_index": j,
                }

                all_ids.append(chunk_id)
                all_documents.append(chunk)
                all_metadatas.append(chunk_metadata)

        self.documents_collection.add(
            ids=all_ids, documents=all_documents, metadatas=all_metadatas
        )

        self.logger.info(f"{len(texts)} belge eklendi: {len(all_documents)} chunk")
        return all_ids

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))

        return chunks if chunks else [text]

    def search_documents(
        self, query: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        results = self.documents_collection.query(
            query_texts=[query], n_results=n_results, where=where
        )

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }

    def add_conversation(
        self,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        conv_id = f"conv_{len(self.conversation_history)}"

        combined_text = f"Kullanıcı: {user_message}\nAsistan: {assistant_response}"

        self.conversations_collection.add(
            ids=[conv_id], documents=[combined_text], metadatas=[metadata or {}]
        )

        self.conversation_history.append(
            {"user": user_message, "assistant": assistant_response}
        )

        if len(self.conversation_history) > self.max_history:
            old_id = f"conv_{len(self.conversation_history) - self.max_history - 1}"
            try:
                self.conversations_collection.delete(ids=[old_id])
            except:
                pass

    def search_conversations(
        self, query: str, n_results: int = 3
    ) -> List[Dict[str, str]]:
        results = self.conversations_collection.query(
            query_texts=[query], n_results=n_results
        )

        return results["documents"][0] if results["documents"] else []

    def get_relevant_context(
        self, query: str, n_docs: int = 3, n_convs: int = 2
    ) -> str:
        doc_results = self.search_documents(query, n_results=n_docs)
        conv_results = self.search_conversations(query, n_results=n_convs)

        context_parts = []

        if doc_results["documents"]:
            context_parts.append("İlgili Belgeler:")
            for doc in doc_results["documents"]:
                context_parts.append(f"- {doc}")

        if conv_results:
            context_parts.append("\nİlgili Konuşmalar:")
            for conv in conv_results:
                context_parts.append(f"- {conv}")

        return "\n".join(context_parts)

    def clear_documents(self) -> None:
        self.chroma_client.delete_collection("documents")
        self.documents_collection = self.chroma_client.get_or_create_collection(
            name="documents", embedding_function=self.embedding_function
        )
        self.logger.info("Belgeler temizlendi")

    def clear_conversations(self) -> None:
        self.chroma_client.delete_collection("conversations")
        self.conversations_collection = self.chroma_client.get_or_create_collection(
            name="conversations", embedding_function=self.embedding_function
        )
        self.conversation_history = []
        self.logger.info("Konuşmalar temizlendi")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "document_count": self.documents_collection.count(),
            "conversation_count": self.conversations_collection.count(),
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
        }
