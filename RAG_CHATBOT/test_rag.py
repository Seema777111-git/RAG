from query_processor import get_context
from llm import generate_answer


question = "Why do some students score well without studying constantly?"

context = get_context(question)

answer = generate_answer(question, context)

print("\n--- FINAL ANSWER ---")
print(answer)