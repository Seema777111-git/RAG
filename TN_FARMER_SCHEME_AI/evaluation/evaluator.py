import json
from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from rag import load_rag_components, run_rag


# ============================================================
# FILE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_FILE = BASE_DIR / "questions.json"


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"questions.json not found at:\n"
            f"{QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "questions.json must contain a list."
        )

    return questions


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(question, components):

    question_id = question.get("id")
    question_type = question.get("type")
    language = question.get("language")
    question_text = question.get("question")

    result = run_rag(
        query=question_text,
        components=components
    )

    question_status = result.get(
        "question_status",
        ""
    )

    evidence = result.get(
        "evidence",
        []
    )

    generated = bool(
        result.get(
            "generated",
            False
        )
    )

    # --------------------------------------------------------
    # VALID QUESTION
    # --------------------------------------------------------

    if question_type == "valid":

        if generated and evidence:

            status = "PASS"

            reason = (
                "Valid question reached the generator "
                "with official evidence."
            )

        else:

            status = "FAIL"

            reason = (
                "Valid question did not produce "
                "usable official evidence."
            )

    # --------------------------------------------------------
    # INSUFFICIENT QUESTION
    # --------------------------------------------------------

    elif question_type == "insufficient":

        if (
            not generated
            and question_status == "INSUFFICIENT"
        ):

            status = "PASS"

            reason = (
                "Insufficient question was correctly "
                "blocked."
            )

        else:

            status = "FAIL"

            reason = (
                "Insufficient question was not "
                "correctly blocked."
            )

    # --------------------------------------------------------
    # IRRELEVANT QUESTION
    # --------------------------------------------------------

    elif question_type == "irrelevant":

        if (
            not generated
            and question_status == "IRRELEVANT"
        ):

            status = "PASS"

            reason = (
                "Irrelevant question was correctly "
                "blocked."
            )

        else:

            status = "FAIL"

            reason = (
                "Irrelevant question was not "
                "correctly blocked."
            )

    # --------------------------------------------------------
    # UNKNOWN TYPE
    # --------------------------------------------------------

    else:

        status = "FAIL"

        reason = (
            f"Unknown question type: "
            f"{question_type}"
        )

    return {
        "id": question_id,
        "type": question_type,
        "language": language,
        "question": question_text,
        "question_status": question_status,
        "generated": generated,
        "evidence_count": len(evidence),
        "status": status,
        "reason": reason,
        "answer": result.get(
            "answer",
            ""
        )
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(result):

    print("\n-----------------------------------")

    print(
        f"TEST {result['id']}"
    )

    print(
        "Type:",
        result["type"]
    )

    print(
        "Language:",
        result["language"]
    )

    print(
        "Question:",
        result["question"]
    )

    print(
        "Question Status:",
        result["question_status"]
    )

    print(
        "Generated:",
        result["generated"]
    )

    print(
        "Evidence Count:",
        result["evidence_count"]
    )

    print(
        "RESULT:",
        result["status"]
    )

    print(
        "Reason:",
        result["reason"]
    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(results):

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["status"] == "PASS"
    )

    failed = total - passed

    if total > 0:
        pass_rate = (
            passed / total
        ) * 100
    else:
        pass_rate = 0

    print("\n===================================")
    print("EVALUATION SUMMARY")
    print("===================================")

    print(
        "\nTotal tests :",
        total
    )

    print(
        "Passed      :",
        passed
    )

    print(
        "Failed      :",
        failed
    )

    print(
        "Pass rate   :",
        f"{pass_rate:.2f}%"
    )

    print("\n===================================")

    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")

    print("===================================")


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():

    print("===================================")
    print("TN FARMER SCHEME AI")
    print("RAG EVALUATOR")
    print("===================================")

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    print("\n[1] Loading test questions...")

    questions = load_questions()

    print(
        "Test questions:",
        len(questions)
    )

    # --------------------------------------------------------
    # LOAD RAG COMPONENTS ONCE
    # --------------------------------------------------------

    print(
        "\n[2] Loading RAG components..."
    )

    components = load_rag_components()

    # --------------------------------------------------------
    # RUN ALL TESTS
    # --------------------------------------------------------

    print(
        "\n[3] Running evaluation..."
    )

    results = []

    for question in questions:

        result = evaluate_question(
            question=question,
            components=components
        )

        results.append(result)

        display_result(result)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    display_summary(results)

    return results


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    run_evaluation()