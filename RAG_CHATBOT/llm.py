from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Load API key from .env
load_dotenv()


def generate_answer(question, context):
    """
    Generate an answer using the retrieved context.
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = f"""
You are a helpful parenting assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content