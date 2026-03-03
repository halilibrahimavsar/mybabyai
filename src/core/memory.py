import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import time
import math
from datetime import datetime

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
        
        self.decay_lambda = self.config.get("memory.decay_lambda", 0.01) # lambda param for exponential decay

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
            chunk_metadata["created_at"] = time.time()
            chunk_metadata["last_accessed"] = time.time()
            chunk_metadata["access_count"] = 0

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
        # Fetch more to apply decay calculation before truncating
        fetch_count = n_results * 3
        results = self.documents_collection.query(
            query_texts=[query], n_results=fetch_count, where=where
        )

        if not results["ids"] or not results["ids"][0]:
             return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        # Apply Nisyan (Decay) formula
        current_time = time.time()
        scored_results = []
        
        ids_to_update = []
        metadatas_to_update = []

        for j in range(len(results["ids"][0])):
            doc_id = results["ids"][0][j]
            dist = results["distances"][0][j]
            meta = results["metadatas"][0][j]
            doc = results["documents"][0][j]
            
            # Subtly increase distance based on age (larger distance = less relevant in chroma typically)
            created_at = meta.get("created_at", current_time)
            age_hours = (current_time - created_at) / 3600.0
            
            # Decay formula: decayed_distance = distance * e^(lambda * age_hours)
            decayed_dist = dist * math.exp(self.decay_lambda * age_hours)
            
            # Bonus for frequent access
            access_count = meta.get("access_count", 0)
            if access_count > 0:
                 decayed_dist = decayed_dist / (1.0 + 0.1 * math.log(1 + access_count))
                 
            scored_results.append((decayed_dist, doc_id, doc, meta))
            
        # Sort by decayed distance (ascending, so smaller distance first)
        scored_results.sort(key=lambda x: x[0])
        
        # Take top n_results
        top_results = scored_results[:n_results]
        
        # Update access stats for retrieved documents
        for _, doc_id, _, meta in top_results:
             meta["last_accessed"] = current_time
             meta["access_count"] = meta.get("access_count", 0) + 1
             ids_to_update.append(doc_id)
             metadatas_to_update.append(meta)
             
        if ids_to_update:
             # Find corresponding documents to pass to update
             updated_docs = [r[2] for r in top_results]
             self.documents_collection.update(ids=ids_to_update, metadatas=metadatas_to_update, documents=updated_docs)

        return {
            "ids": [r[1] for r in top_results],
            "documents": [r[2] for r in top_results],
            "metadatas": [r[3] for r in top_results],
            "distances": [r[0] for r in top_results],
        }

    def add_conversation(
        self,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        conv_id = f"conv_{len(self.conversation_history)}"

        combined_text = f"Kullanıcı: {user_message}\nAsistan: {assistant_response}"
        
        final_meta = metadata or {}
        final_meta["created_at"] = time.time()
        final_meta["last_accessed"] = time.time()
        final_meta["access_count"] = 0

        self.conversations_collection.add(
            ids=[conv_id], documents=[combined_text], metadatas=[final_meta]
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
        
    def zikir(self, top_k: int = 100) -> None:
        """
        Replay / Consolidation loop (Zikir).
        Gathers highly accessed memories and refreshes their created_at time 
        to combat decay, cementing them as core knowledge.
        """
        self.logger.info("Zikir (Memory Consolidation) başlatılıyor...")
        
        try:
             # We unfortunately can't easily query "order by access_count" in default Chroma.
             # Fetch a large chunk.
             all_data = self.documents_collection.get()
             if not all_data or not all_data['ids']:
                  return
                  
             ids = all_data['ids']
             metadatas = all_data['metadatas']
             
             # Sort by access count
             items = list(zip(ids, metadatas))
             items.sort(key=lambda x: x[1].get('access_count', 0), reverse=True)
             
             # Take top_k
             top_items = items[:top_k]
             
             ids_to_update = []
             metas_to_update = []
             current_time = time.time()
             
             for doc_id, meta in top_items:
                  if meta.get('access_count', 0) > 2: # Only consolidate if actually used
                       # Reset creation time to essentially negate decay
                       meta['created_at'] = current_time
                       # Reset access count slightly to avoid exponential inflation?
                       # We leave it for now to let them stay strong.
                       ids_to_update.append(doc_id)
                       metas_to_update.append(meta)
                       
             if ids_to_update:
                  self.documents_collection.update(ids=ids_to_update, metadatas=metas_to_update)
                  self.logger.info(f"Zikir tamamlandı. {len(ids_to_update)} hafıza parçası pekiştirildi.")
                  
        except Exception as e:
             self.logger.error(f"Zikir işlemi sırasında hata oluştu: {e}")
