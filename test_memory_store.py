
from app.memory_store import store_memory, search_memories

user_id = "test-user-001"
session_id = "a1b2c3d4-e5f6-4789-8123-456789abcdef"

memory_id = store_memory(
    user_id=user_id,
    session_id=session_id,
    memory_text="The employee vacation policy recommends 10 business days of paid vacation per year.",
)

print("Stored memory ID:", memory_id)

results = search_memories(
    user_id=user_id,
    query="How many paid vacation days are recommended?",
)

for result in results:
    print(result)
