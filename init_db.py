
from app.db.session import Base, engine
from app.db.models import DocumentChunk, ConversationMemory


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    init_db()
