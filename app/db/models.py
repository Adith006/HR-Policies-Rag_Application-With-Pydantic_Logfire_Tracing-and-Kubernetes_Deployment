
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)

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


class ChatConversation(Base):
    """Stores a conversation and its ownership information."""

    __tablename__ = "chat_conversations"

    id = Column(String(36), primary_key=True)

    user_id = Column(String(255), nullable=False, index=True)

    title = Column(
        String(255),
        nullable=False,
        default="New conversation",
    )

    is_pinned = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ChatMessage(Base):
    """Stores individual user and assistant messages."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)

    conversation_id = Column(
        String(36),
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    role = Column(String(20), nullable=False)

    content = Column(Text, nullable=False)

    metadata_ = Column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
