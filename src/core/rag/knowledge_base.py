"""
RAG Knowledge Base for CodeMind

Provides retrieval-augmented generation by:
1. Storing training data in ChromaDB
2. Retrieving relevant context for queries
3. Augmenting prompts with retrieved context
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from src.core.prompting import TOKENS, build_instruction_prompt

ROOT = Path(__file__).parent.parent


class KnowledgeBase:
    def __init__(
        self,
        db_path: str = "codemind/data/knowledge_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )

        self.client = chromadb.PersistentClient(
            path=str(self.db_path), settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="codemind_knowledge",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def _generate_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def add_instruction(
        self,
        instruction: str,
        response: str,
        language: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        combined = f"Q: {instruction}\nA: {response}"
        doc_id = self._generate_id(combined)

        full_metadata = {
            "language": language,
            "type": "instruction",
            **(metadata or {}),
        }

        self.collection.upsert(
            ids=[doc_id], documents=[combined], metadatas=[full_metadata]
        )

        return doc_id

    def add_batch(
        self,
        instructions: List[str],
        responses: List[str],
        languages: Optional[List[str]] = None,
    ) -> List[str]:
        ids = []
        documents = []
        metadatas = []

        for i, (instruction, response) in enumerate(zip(instructions, responses)):
            combined = f"Q: {instruction}\nA: {response}"
            doc_id = self._generate_id(combined)

            ids.append(doc_id)
            documents.append(combined)
            metadatas.append(
                {
                    "language": languages[i] if languages else "general",
                    "type": "instruction",
                    "index": i,
                }
            )

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        return ids

    def load_training_data(self, data_path: str) -> int:
        data_path = Path(data_path)

        if not data_path.exists():
            return 0

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instructions = []
        responses = []
        languages = []

        for item in data:
            instruction = item.get("user", item.get("instruction", ""))
            response = item.get("assistant", item.get("output", ""))
            language = item.get("language", "general")

            if instruction and response:
                instructions.append(instruction)
                responses.append(response)
                languages.append(language)

        if instructions:
            self.add_batch(instructions, responses, languages)

        return len(instructions)

    def search(
        self,
        query: str,
        n_results: int = 5,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        where_filter = None
        if language:
            where_filter = {"language": language}

        results = self.collection.query(
            query_texts=[query], n_results=n_results, where=where_filter
        )

        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append(
                    {
                        "document": doc,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "distance": results["distances"][0][i]
                        if results["distances"]
                        else 0,
                    }
                )

        return formatted

    def get_context(
        self,
        query: str,
        n_results: int = 3,
        max_tokens: int = 500,
    ) -> str:
        results = self.search(query, n_results=n_results)

        if not results:
            return ""

        context_parts = [TOKENS.context]
        current_tokens = 0

        for result in results:
            doc = result["document"]
            tokens = len(doc.split())

            if current_tokens + tokens > max_tokens:
                break

            context_parts.append(doc)
            current_tokens += tokens

        context_parts.append(TOKENS.context_end)

        return "\n".join(context_parts)

    def build_rag_prompt(
        self,
        query: str,
        n_context: int = 3,
        max_context_tokens: int = 500,
    ) -> str:
        context = self.get_context(
            query, n_results=n_context, max_tokens=max_context_tokens
        )

        if context:
            base = build_instruction_prompt(
                user=query, assistant=None, language="general", include_eos=False
            )
            return f"{context}\n\n{base}"
        return build_instruction_prompt(
            user=query, assistant=None, language="general", include_eos=False
        )

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection("codemind_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="codemind_knowledge",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
