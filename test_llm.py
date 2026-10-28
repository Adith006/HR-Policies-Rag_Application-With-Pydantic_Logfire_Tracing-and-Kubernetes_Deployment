from app.llm.client import generate_answer

answer = generate_answer(
    question="Who should an employee report workplace harassment to?",
    context=(
        "An employee should report workplace harassment to the "
        "Executive Director, Chair, Vice Chair, or Treasurer of "
        "the Board of Directors."
    ),
)

print("Generated answer:")
print(answer)
