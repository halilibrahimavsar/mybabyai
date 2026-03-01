import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

import sys


from src.utils.config import Config
from src.utils.logger import get_logger


Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), default="Yeni Sohbet")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column(Text, nullable=True)
    indexed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    status = Column(String(50), default="pending")
    config = Column(Text, nullable=True)
    metrics = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Database:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_logger("database")

        self.base_dir = self.config.base_dir
        self.db_path = self.config.get_path("database.path", "data/mybabyai.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

        self._create_tables()

    def _create_tables(self) -> None:
        Base.metadata.create_all(self.engine)
        self.logger.info("Veritabanı tabloları oluşturuldu")

    def get_session(self) -> Session:
        return self.SessionLocal()

    def create_conversation(self, title: str = "Yeni Sohbet") -> Conversation:
        with self.get_session() as session:
            conv = Conversation(title=title)
            session.add(conv)
            session.commit()
            session.refresh(conv)
            return conv

    def get_conversation(self, conv_id: int) -> Optional[Conversation]:
        with self.get_session() as session:
            return (
                session.query(Conversation).filter(Conversation.id == conv_id).first()
            )

    def get_all_conversations(self, limit: int = 50) -> List[Conversation]:
        with self.get_session() as session:
            return (
                session.query(Conversation)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
                .all()
            )

    def delete_conversation(self, conv_id: int) -> bool:
        with self.get_session() as session:
            conv = (
                session.query(Conversation).filter(Conversation.id == conv_id).first()
            )
            if conv:
                session.delete(conv)
                session.commit()
                return True
            return False

    def add_message(self, conversation_id: int, role: str, content: str) -> Message:
        with self.get_session() as session:
            msg = Message(conversation_id=conversation_id, role=role, content=content)
            session.add(msg)
            session.commit()
            session.refresh(msg)
            return msg

    def get_messages(self, conversation_id: int) -> List[Message]:
        with self.get_session() as session:
            return (
                session.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
                .all()
            )

    def add_document(
        self,
        filename: str,
        content: str,
        filepath: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Document:
        with self.get_session() as session:
            doc = Document(
                filename=filename,
                filepath=filepath,
                content=content,
                doc_metadata=json.dumps(metadata) if metadata else None,
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc

    def get_all_documents(self) -> List[Document]:
        with self.get_session() as session:
            return session.query(Document).order_by(Document.created_at.desc()).all()

    def delete_document(self, doc_id: int) -> bool:
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                session.delete(doc)
                session.commit()
                return True
            return False

    def create_training_job(
        self, name: str, config: Optional[Dict] = None
    ) -> TrainingJob:
        with self.get_session() as session:
            job = TrainingJob(name=name, config=json.dumps(config) if config else None)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def update_training_job(
        self, job_id: int, status: Optional[str] = None, metrics: Optional[Dict] = None
    ) -> Optional[TrainingJob]:
        with self.get_session() as session:
            job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                if status:
                    job.status = status
                    if status == "running":
                        job.started_at = datetime.utcnow()
                    elif status in ["completed", "failed"]:
                        job.completed_at = datetime.utcnow()
                if metrics:
                    job.metrics = json.dumps(metrics)
                session.commit()
                session.refresh(job)
                return job
            return None

    def get_training_jobs(self) -> List[TrainingJob]:
        with self.get_session() as session:
            return (
                session.query(TrainingJob).order_by(TrainingJob.created_at.desc()).all()
            )

    def export_conversation(self, conv_id: int, format: str = "json") -> str:
        conv = self.get_conversation(conv_id)
        if not conv:
            return ""

        messages = self.get_messages(conv_id)

        if format == "json":
            data = {
                "title": conv.title,
                "created_at": str(conv.created_at),
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "created_at": str(m.created_at),
                    }
                    for m in messages
                ],
            }
            return json.dumps(data, ensure_ascii=False, indent=2)

        elif format == "txt":
            lines = [f"# {conv.title}", f"# {conv.created_at}", ""]
            for m in messages:
                lines.append(f"[{m.role.upper()}]: {m.content}")
                lines.append("")
            return "\n".join(lines)

        return ""
