import json
import re
import requests


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "qwen2.5:3b"

# Maximum number of tokens Qwen can generate.
# This is a ceiling, not a target.
NUM_PREDICT = 300

# Lower temperature = more factual and consistent.
TEMPERATURE = 0.1

# Keep model loaded in Ollama memory for faster follow-up questions.
KEEP_ALIVE = "10m"

# Request timeout.
TIMEOUT = 180

# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_query_language(query: str) -> str:
    """
    Lightweight heuristic: Tamil script vs Latin script.
    Good enough to steer the LLM; doesn't need to be perfect.
    """
    tamil_chars = re.findall(r'[\u0B80-\u0BFF]', query)
    if len(tamil_chars) > 2:
        return "Tamil"
    return "English"

# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(query, context):
    """
    Build a concise grounded prompt for Qwen.

    Qwen must answer ONLY from the supplied official evidence.
    """

    answer_language = detect_query_language(query)

    prompt = f"""
You are TN Farmer Scheme AI Assistant for the Government of Tamil Nadu.

Answer the farmer's question using ONLY the official evidence provided below.

IMPORTANT RULES:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the evidence does not contain the answer, clearly say that the
   available official information does not provide the answer.
4. Keep the answer concise and easy for a farmer to understand.
5. Include the benefit/subsidy when available.
6. Include eligibility or documents only when relevant to the question.
7. Do not repeat the same information.
8. Do not mention that you are an AI unless necessary.
9. Do not mention internal RAG, embeddings, FAISS, reranking or models.
10. Do not create amounts, percentages, dates or eligibility conditions
    that are not present in the evidence.
11. Never repeat the farmer's question back as the answer.
12. Never repeat the same sentence, phrase, or word more than once.
13. Write the answer only once. Do not continue after you have
    finished answering.

FARMER QUESTION:
{query}

OFFICIAL EVIDENCE:
{context}

LANGUAGE INSTRUCTION (MUST FOLLOW):
Respond ONLY in {answer_language}, matching the language of the FARMER
QUESTION above — even if the OFFICIAL EVIDENCE above is written in a
different language. Do not switch languages partway through the answer.

Now provide the most useful concise answer, in {answer_language}, based
strictly on the official evidence.
"""

    return prompt.strip()


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    """
    Check whether Ollama is running.
    """

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )

        if response.status_code == 200:
            return True

        return False

    except requests.RequestException:
        return False


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(query, context):
    """
    Generate an answer using Qwen through Ollama.

    Ollama streaming is enabled internally so Qwen can begin
    generating immediately. The complete response is collected
    before returning because the existing RAG pipeline expects
    a normal string.
    """

    if not check_ollama():

        return (
            "Ollama is not running. "
            "Please start Ollama and try again."
        )

    prompt = build_prompt(
        query=query,
        context=context
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "repeat_penalty": 1.3,
            "repeat_last_n": 128,
            "top_p": 0.9,
            "top_k": 40
        }
    }


    try:

        print("\nSending request to Ollama...")
        print("Model:", MODEL_NAME)
        print("Maximum output tokens:", NUM_PREDICT)

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        answer_parts = []

        for line in response.iter_lines(
            decode_unicode=True
        ):

            if not line:
                continue

            try:

                data = json.loads(line)

                chunk = data.get(
                    "response",
                    ""
                )

                if chunk:
                    answer_parts.append(chunk)

                if data.get(
                    "done",
                    False
                ):
                    break

            except json.JSONDecodeError:

                continue

        answer = "".join(
            answer_parts
        ).strip()

        if not answer:

            return (
                "I could not generate an answer from "
                "the available official evidence."
            )

        print("Qwen response received.")

        return answer

    except requests.Timeout:

        return (
            "The AI response is taking too long. "
            "Please try the question again."
        )

    except requests.RequestException as error:

        print("Ollama error:", error)

        return (
            "Unable to connect to the local AI model. "
            "Please make sure Ollama is running."
        )

    except Exception as error:

        print("Generation error:", error)

        return (
            "An error occurred while generating the answer."
        )


# ============================================================
# GENERATE FROM GUARDRAIL EVIDENCE
# ============================================================

def generate_from_evidence(
    query,
    guardrail_result
):
    """
    Generate an answer from approved guardrail evidence.

    This interface is intentionally kept compatible with rag.py.
    """

    if not guardrail_result:

        return {
            "answer": (
                "No evidence is available "
                "to answer this question."
            ),
            "evidence": [],
            "generated": False
        }

    if not guardrail_result.get(
        "allowed",
        False
    ):

        return {
            "answer": guardrail_result.get(
                "message",
                "This question cannot be answered from the available evidence."
            ),
            "evidence": guardrail_result.get(
                "evidence",
                []
            ),
            "generated": False
        }

    context = guardrail_result.get(
        "context",
        ""
    )

    evidence = guardrail_result.get(
        "evidence",
        []
    )

    if not context:

        return {
            "answer": (
                "No official evidence is available "
                "to answer this question."
            ),
            "evidence": evidence,
            "generated": False
        }

    answer = generate_answer(
        query=query,
        context=context
    )

    # If Ollama failed, don't mark the answer as successfully generated.
    failed_messages = [
        "Ollama is not running.",
        "Unable to connect to the local AI model.",
        "An error occurred while generating the answer.",
        "The AI response is taking too long.",
        "I could not generate an answer"
    ]

    generated = not any(
        message in answer
        for message in failed_messages
    )

    return {
        "answer": answer,
        "evidence": evidence,
        "generated": generated
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TN FARMER SCHEME AI")
    print("GENERATOR - QWEN / OLLAMA")
    print("=" * 70)

    print("\nModel:")
    print(MODEL_NAME)

    print("\nChecking Ollama...")

    if check_ollama():

        print("Ollama is running.")

    else:

        print(
            "Ollama is not running."
        )

        print(
            "\nStart Ollama before testing."
        )

        raise SystemExit

    test_query = (
        "What subsidy is available for farmers "
        "for agricultural machinery?"
    )

    test_context = """
Scheme Name:
Sub-Mission on Agricultural Mechanisation (Agri)

Department:
Agriculture

Eligibility:
All District Farmer

Documents:
Patta, Chitta, Adangal

Official Source:
https://www.tnagrisnet.tn.gov.in/people_app/goScheme/
"""

    print("\nQuestion:")
    print(test_query)

    print("\nGenerating answer...")

    answer = generate_answer(
        query=test_query,
        context=test_context
    )

    print("\n" + "-" * 70)
    print("ANSWER")
    print("-" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETED")
    print("=" * 70)

