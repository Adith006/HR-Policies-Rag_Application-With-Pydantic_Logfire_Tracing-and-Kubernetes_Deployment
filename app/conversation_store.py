
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import func, select

from app.db.models import ChatConversation, ChatMessage
from app.db.session import SessionLocal


def _validate_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    return user_id.strip()


def _serialize_conversation(conversation, message_count=0) -> dict:
    return {
        "id": conversation.id,
        "user_id": conversation.user_id,
        "title": conversation.title,
        "is_pinned": conversation.is_pinned,
        "created_at": (
            conversation.created_at.isoformat()
            if conversation.created_at else None
        ),
        "updated_at": (
            conversation.updated_at.isoformat()
            if conversation.updated_at else None
        ),
        "message_count": message_count,
    }


def create_conversation(
    user_id: str,
    conversation_id: str | None = None,
) -> dict:
    """Create a new persistent conversation."""

    user_id = _validate_user_id(user_id)
    conversation_id = str(UUID(conversation_id)) if conversation_id else str(uuid4())

    with SessionLocal() as db:
        existing = db.get(ChatConversation, conversation_id)

        if existing:
            if existing.user_id != user_id:
                raise PermissionError("Conversation access denied.")

            return _serialize_conversation(existing)

        conversation = ChatConversation(
            id=conversation_id,
            user_id=user_id,
            title="New conversation",
            is_pinned=False,
        )

        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        return _serialize_conversation(conversation)


def save_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> dict:
    """Save one user or assistant message."""

    user_id = _validate_user_id(user_id)

    conversation_id = str(UUID(conversation_id))

    if role not in {"user", "assistant"}:
        raise ValueError("Role must be user or assistant.")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Message content cannot be empty.")

    with SessionLocal() as db:
        conversation = db.execute(
            select(ChatConversation).where(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == user_id,
            )
        ).scalar_one_or_none()

        if conversation is None:
            raise LookupError("Conversation was not found.")

        message = ChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content.strip(),
            metadata_=metadata or {},
        )

        db.add(message)

        # Use the first user question as the conversation title.
        if role == "user" and conversation.title == "New conversation":
            title = " ".join(content.split())
            conversation.title = (
                title[:77] + "..."
                if len(title) > 80
                else title
            )

        conversation.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(message)

        return {
            "id": message.id,
            "conversation_id": message.conversation_id,
            "role": message.role,
            "content": message.content,
            "metadata": message.metadata_,
            "created_at": (
                message.created_at.isoformat()
                if message.created_at else None
            ),
        }


def get_conversations(
    user_id: str,
    limit: int = 50,
) -> list[dict]:
    """Return conversation history with pinned conversations first."""

    user_id = _validate_user_id(user_id)

    if not 1 <= limit <= 200:
        raise ValueError("Limit must be between 1 and 200.")

    with SessionLocal() as db:
        conversations = db.execute(
            select(ChatConversation)
            .where(ChatConversation.user_id == user_id)
            .order_by(
                ChatConversation.is_pinned.desc(),
                ChatConversation.updated_at.desc(),
            )
            .limit(limit)
        ).scalars().all()

        if not conversations:
            return []

        conversation_ids = [item.id for item in conversations]

        counts = db.execute(
            select(
                ChatMessage.conversation_id,
                func.count(ChatMessage.id),
            )
            .where(ChatMessage.conversation_id.in_(conversation_ids))
            .group_by(ChatMessage.conversation_id)
        ).all()

        count_map = {
            conversation_id: count
            for conversation_id, count in counts
        }

        return [
            _serialize_conversation(
                conversation,
                count_map.get(conversation.id, 0),
            )
            for conversation in conversations
        ]


def get_messages(
    conversation_id: str,
    user_id: str,
) -> list[dict]:
    """Load messages after verifying conversation ownership."""

    user_id = _validate_user_id(user_id)
    conversation_id = str(UUID(conversation_id))

    with SessionLocal() as db:
        conversation = db.execute(
            select(ChatConversation.id).where(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == user_id,
            )
        ).scalar_one_or_none()

        if conversation is None:
            raise LookupError("Conversation was not found.")

        messages = db.execute(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(
                ChatMessage.created_at.asc(),
                ChatMessage.id.asc(),
            )
        ).scalars().all()

        return [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "metadata": message.metadata_ or {},
                "created_at": (
                    message.created_at.isoformat()
                    if message.created_at else None
                ),
            }
            for message in messages
        ]


def set_conversation_pinned(
    conversation_id: str,
    user_id: str,
    is_pinned: bool,
) -> bool:
    """Pin or unpin an owned conversation."""

    user_id = _validate_user_id(user_id)
    conversation_id = str(UUID(conversation_id))

    with SessionLocal() as db:
        conversation = db.execute(
            select(ChatConversation).where(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == user_id,
            )
        ).scalar_one_or_none()

        if conversation is None:
            return False

        conversation.is_pinned = bool(is_pinned)
        conversation.updated_at = datetime.now(timezone.utc)

        db.commit()

        return True


def delete_conversation(
    conversation_id: str,
    user_id: str,
) -> bool:
    """Delete an owned conversation and its messages."""

    user_id = _validate_user_id(user_id)
    conversation_id = str(UUID(conversation_id))

    with SessionLocal() as db:
        conversation = db.execute(
            select(ChatConversation).where(
                ChatConversation.id == conversation_id,
                ChatConversation.user_id == user_id,
            )
        ).scalar_one_or_none()

        if conversation is None:
            return False

        db.delete(conversation)
        db.commit()

        return True
