
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from app.db.session import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(255), nullable=False)
    page_number = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)

    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)


class ConversationMemory(Base):
    __tablename__ = "conversation_memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(36), nullable=False, index=True)
    memory_text = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)
    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)
