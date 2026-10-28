import os

from dotenv import load_dotenv
from portkey_ai import Portkey

load_dotenv()

client = Portkey(
    api_key=os.environ["PORTKEY_API_KEY"],
    config="pc-enterp-1a6936",
)
response = client.chat.completions.create(
    model=os.environ["GROQ_MODEL"],
    messages=[
        {
            "role": "user",
            "content": (
                "Answer this question in one sentence: "
                "Who should an employee report workplace harassment to?"
            ),
        }
    ],
    max_tokens=100,
    temperature=0,
)

print("\nFallback test response:")
print(response.choices[0].message.content)

print("\nToken usage:")
print(response.usage)