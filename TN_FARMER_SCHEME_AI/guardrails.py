import re
from urllib.parse import urlparse


# ============================================================
# CONFIGURATION
# ============================================================

MIN_RESULTS_REQUIRED = 1
MAX_EVIDENCE = 3

# These act only as a basic sanity floor, not the primary
# relevance filter. The cross-encoder reranker is English-only
# and scores Tamil/Tanglish text unreliably (even genuine
# on-topic Tanglish questions can score deeply negative), so
# domain relevance is primarily enforced via keyword matching
# in is_agriculture_related(), not via these score thresholds.
#
# Values tuned from test_thresholds.py output on 2026-09-08:
#   - Legitimate on-topic Tanglish question scored as low as
#     retrieval=0.65 / rerank=-10.5
#   - Off-topic "capital of France" scored retrieval=0.37
#   - So thresholds are set just below the weakest legitimate
#     case, mainly to catch obviously nonsensical matches.
MIN_RERANK_SCORE = -12.0
MIN_RETRIEVAL_SCORE = 0.45

OFFICIAL_DOMAIN = "tnagrisnet.tn.gov.in"


# ============================================================
# DOMAIN KEYWORDS
# ============================================================

# English keywords related to Tamil Nadu agriculture schemes.
#
# NOTE: Generic words like "apply", "document", "benefit",
# "eligibility", "assistance", "guideline" were intentionally
# REMOVED from this list. They are not agriculture-specific
# (e.g. "How do I apply for a passport?" was matching via
# "apply") and caused off-topic questions to be misclassified
# as agriculture-related before ever reaching retrieval.
ENGLISH_DOMAIN_KEYWORDS = [
    "scheme",
    "schemes",
    "subsidy",
    "subsidies",
    "farmer",
    "farmers",
    "agriculture",
    "agricultural",
    "farming",
    "crop",
    "crops",
    "seed",
    "seeds",
    "fertilizer",
    "fertiliser",
    "irrigation",
    "machinery",
    "machine",
    "tractor",
    "equipment",
    "horticulture",
    "organic",
    "soil",
    "pesticide",
    "drip",
    "sprinkler",
    "cultivation",
    "farm",
    "agri",
    "agriculture department",
    "government scheme",
    "government schemes",
    "tn scheme",
    "tamil nadu scheme",
    "tamil nadu schemes"
]


# Tamil keywords related to agriculture schemes
TAMIL_DOMAIN_KEYWORDS = [
    "விவசாயம்",
    "விவசாயி",
    "விவசாயிகள்",
    "வேளாண்மை",
    "வேளாண்",
    "திட்டம்",
    "திட்டங்கள்",
    "மானியம்",
    "மாநியம்",
    "உதவி",
    "விதை",
    "விதைகள்",
    "உரம்",
    "பாசனம்",
    "இயந்திரம்",
    "இயந்திரங்கள்",
    "டிராக்டர்",
    "தோட்டக்கலை",
    "இயற்கை விவசாயம்",
    "மண்",
    "பயிர்",
    "பயிர்கள்",
    "பூச்சிக்கொல்லி",
    "தகுதி",
    "ஆவணம்",
    "ஆவணங்கள்",
    "விண்ணப்பம்",
    "விண்ணப்பிக்க",
    "நன்மை",
    "நன்மைகள்",
    "வழிகாட்டுதல்",
    "வேளாண் திட்டம்",
    "அரசு திட்டம்",
    "அரசு திட்டங்கள்"
]


# Tanglish keywords
TANGLISH_DOMAIN_KEYWORDS = [
    "vivasaayam",
    "vivasaayi",
    "vivasaayigal",
    "velaanmai",
    "velaan",
    "thittam",
    "thittangal",
    "maanayam",
    "maaniyam",
    "udhavi",
    "vidhai",
    "vidhaigal",
    "uram",
    "pasanam",
    "iyandhiram",
    "iyandhirangal",
    "tractor",
    "thottakkalai",
    "iyarkai vivasayam",
    "mann",
    "payir",
    "payirgal",
    "poochikkolli",
    "thaguthi",
    "aavanam",
    "aavanangal",
    "vinnappam",
    "vinnappikka",
    "nanmai",
    "nanmaigal",
    "vazikaattuthal",
    "eligibility",
    "irrigation"
]


# ============================================================
# QUESTION VALIDATION
# ============================================================

def validate_question(query):
    """
    Validate the farmer's question.

    Returns:
        (True, "") if the question can proceed.
        (False, message) if the question should be stopped.
    """

    if query is None:
        return False, "Please enter a question."

    query = str(query).strip()

    if not query:
        return False, "Please enter a question."

    if len(query) < 3:
        return False, "Please enter a more detailed question."

    return True, ""


# ============================================================
# QUESTION DOMAIN CHECK
# ============================================================

def is_agriculture_related(query):
    """
    Basic domain check.

    The purpose is not to understand the complete meaning
    of the question. It only checks whether there is a
    reasonable indication that the question is related to
    Tamil Nadu agricultural schemes.
    """

    query_lower = query.lower()

    all_keywords = (
        ENGLISH_DOMAIN_KEYWORDS
        + TAMIL_DOMAIN_KEYWORDS
        + TANGLISH_DOMAIN_KEYWORDS
    )

    for keyword in all_keywords:
        if keyword.lower() in query_lower:
            return True

    return False


def is_insufficient_question(query):
    """
    Detect questions that are too vague to search meaningfully.

    Examples:
        "Subsidy?"
        "Scheme?"
        "Benefits?"
        "Eligibility?"
        "Apply?"
    """

    query_clean = query.strip()

    # Remove common punctuation
    query_clean = re.sub(r"[?!.:,;]+", " ", query_clean)

    # Normalize spaces
    query_clean = " ".join(query_clean.split())

    words = query_clean.split()

    # Very short questions are usually insufficient.
    if len(words) <= 2:
        return True

    # Common vague questions
    vague_questions = [
        "subsidy",
        "subsidies",
        "scheme",
        "schemes",
        "benefit",
        "benefits",
        "eligibility",
        "eligible",
        "documents",
        "document",
        "apply",
        "application",
        "assistance",
        "help",
        "details",
        "information",
        "மானியம்",
        "மாநியம்",
        "திட்டம்",
        "திட்டங்கள்",
        "உதவி",
        "தகுதி",
        "ஆவணம்",
        "ஆவணங்கள்",
        "விண்ணப்பம்",
        "நன்மை",
        "நன்மைகள்"
    ]

    if query_clean.lower() in vague_questions:
        return True

    return False


def get_question_status(query):
    """
    Determine whether the question is:

        VALID
        IRRELEVANT
        INSUFFICIENT

    This is intentionally simple and transparent.
    """

    valid, message = validate_question(query)

    if not valid:
        return {
            "status": "INSUFFICIENT",
            "message": message
        }

    query = str(query).strip()

    # Check vague / insufficient questions first.
    if is_insufficient_question(query):
        return {
            "status": "INSUFFICIENT",
            "message": (
                "Please provide a little more detail. "
                "For example, ask about a specific subsidy, "
                "crop, agricultural machinery, irrigation, "
                "eligibility, documents, or application process."
            )
        }

    # Then check whether the question belongs to our domain.
    if not is_agriculture_related(query):
        return {
            "status": "IRRELEVANT",
            "message": (
                "I can help only with Tamil Nadu Government "
                "agricultural schemes, subsidies, farmer benefits, "
                "eligibility, documents, and application information."
            )
        }

    return {
        "status": "VALID",
        "message": ""
    }


# ============================================================
# OFFICIAL SOURCE VALIDATION
# ============================================================

def is_official_source(source_url):
    """
    Check whether the source belongs to the official
    Tamil Nadu Agrisnet government domain.
    """

    if not source_url:
        return False

    try:
        parsed_url = urlparse(source_url)

        hostname = parsed_url.hostname

        if not hostname:
            return False

        hostname = hostname.lower()

        return (
            hostname == OFFICIAL_DOMAIN
            or hostname.endswith("." + OFFICIAL_DOMAIN)
        )

    except Exception:
        return False


# ============================================================
# EVIDENCE CLEANING
# ============================================================

def clean_evidence_text(text):
    """
    Normalize evidence text before sending it to the LLM.
    """

    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# RESULT VALIDATION
# ============================================================

def validate_result(result):
    """
    Validate one retrieved/reranked result.
    """

    if not isinstance(result, dict):
        return False

    text = result.get("text", "")

    if not text:
        return False

    text = clean_evidence_text(text)

    if not text:
        return False

    metadata = result.get("metadata", {})

    scheme_name = metadata.get("scheme_name", "")

    if not scheme_name:
        return False

    # --------------------------------------------------------
    # RETRIEVAL SCORE GATE (sanity floor only)
    # --------------------------------------------------------

    retrieval_score = result.get("score", None)

    if retrieval_score is not None:

        try:
            retrieval_score = float(retrieval_score)

            if retrieval_score < MIN_RETRIEVAL_SCORE:
                return False

        except (TypeError, ValueError):
            pass

    # --------------------------------------------------------
    # RERANK SCORE GATE (sanity floor only)
    # --------------------------------------------------------

    rerank_score = result.get("rerank_score", None)

    if rerank_score is not None:

        try:
            rerank_score = float(rerank_score)

            if rerank_score < MIN_RERANK_SCORE:
                return False

        except (TypeError, ValueError):
            pass

    return True


# ============================================================
# PREPARE OFFICIAL EVIDENCE
# ============================================================

def prepare_evidence(results, max_results=MAX_EVIDENCE):

    if not results:
        return {
            "allowed": False,
            "message": (
                "No relevant government scheme "
                "information was found."
            ),
            "evidence": []
        }

    evidence = []

    for result in results:

        # Validate result structure
        if not validate_result(result):
            continue

        metadata = result.get("metadata", {})

        text = clean_evidence_text(
            result.get("text", "")
        )

        source_url = metadata.get(
            "source_url",
            ""
        )

        # ----------------------------------------------------
        # OFFICIAL SOURCE CHECK
        # ----------------------------------------------------

        official = is_official_source(source_url)

        # Do not allow non-official evidence into the LLM.
        if not official:
            continue

        # ----------------------------------------------------
        # CREATE EVIDENCE OBJECT
        # ----------------------------------------------------

        evidence_item = {
            "scheme_id": metadata.get(
                "id",
                ""
            ),

            "scheme_name": metadata.get(
                "scheme_name",
                ""
            ),

            "department": metadata.get(
                "department",
                ""
            ),

            "text": text,

            "source_url": source_url,

            "official_source": official,

            "chunk_number": metadata.get(
                "chunk_number",
                ""
            ),

            "retrieval_score": result.get(
                "score",
                0
            ),

            "rerank_score": result.get(
                "rerank_score",
                0
            )
        }

        evidence.append(evidence_item)

        if len(evidence) >= max_results:
            break

    # --------------------------------------------------------
    # NO VALID OFFICIAL EVIDENCE
    # --------------------------------------------------------

    if len(evidence) < MIN_RESULTS_REQUIRED:

        return {
            "allowed": False,
            "message": (
                "The available official Tamil Nadu Government "
                "scheme information does not contain enough "
                "relevant evidence to answer this question."
            ),
            "evidence": []
        }

    return {
        "allowed": True,
        "message": "",
        "evidence": evidence
    }


# ============================================================
# BUILD LLM CONTEXT
# ============================================================

def build_context(evidence):

    context_parts = []

    for number, item in enumerate(
        evidence,
        start=1
    ):

        context = f"""
EVIDENCE {number}

Scheme Name:
{item["scheme_name"]}

Department:
{item["department"]}

Information:
{item["text"]}

Official Source:
{item["source_url"]}
""".strip()

        context_parts.append(context)

    return "\n\n".join(context_parts)


# ============================================================
# MAIN GUARDRAIL FUNCTION
# ============================================================

def apply_guardrails(query, results):

    # --------------------------------------------------------
    # GATE 1 + GATE 2
    # QUESTION VALIDATION
    # --------------------------------------------------------

    question_status = get_question_status(query)

    status = question_status["status"]

    if status != "VALID":

        return {
            "allowed": False,
            "message": question_status["message"],
            "question_status": status,
            "evidence": [],
            "context": ""
        }

    # --------------------------------------------------------
    # GATE 3
    # EVIDENCE VALIDATION
    # --------------------------------------------------------

    evidence_result = prepare_evidence(
        results,
        max_results=MAX_EVIDENCE
    )

    if not evidence_result["allowed"]:

        return {
            "allowed": False,
            "message": evidence_result["message"],
            "question_status": "VALID",
            "evidence": [],
            "context": ""
        }

    # --------------------------------------------------------
    # BUILD CONTROLLED CONTEXT
    # --------------------------------------------------------

    evidence = evidence_result["evidence"]

    context = build_context(evidence)

    return {
        "allowed": True,
        "message": "",
        "question_status": "VALID",
        "evidence": evidence,
        "context": context
    }


# ============================================================
# DISPLAY GUARDRAIL RESULT
# ============================================================

def display_guardrail_result(result):

    print("\n===================================")
    print("GUARDRAIL RESULT")
    print("===================================")

    print(
        "\nAllowed:",
        result.get("allowed")
    )

    print(
        "Question Status:",
        result.get(
            "question_status",
            ""
        )
    )

    # --------------------------------------------------------
    # BLOCKED QUESTION
    # --------------------------------------------------------

    if not result.get("allowed"):

        print(
            "\nMessage:",
            result.get(
                "message",
                ""
            )
        )

        return

    # --------------------------------------------------------
    # VALID EVIDENCE
    # --------------------------------------------------------

    evidence = result.get(
        "evidence",
        []
    )

    print(
        "\nEvidence count:",
        len(evidence)
    )

    print("\n===================================")
    print("EVIDENCE")
    print("===================================")

    for number, item in enumerate(
        evidence,
        start=1
    ):

        print(
            f"\nEVIDENCE {number}"
        )

        print("-----------------------------------")

        print(
            "Scheme:",
            item["scheme_name"]
        )

        print(
            "Department:",
            item["department"]
        )

        print(
            "Official Source:",
            item["official_source"]
        )

        print(
            "Retrieval Score:",
            round(
                float(
                    item["retrieval_score"]
                ),
                4
            )
        )

        print(
            "Rerank Score:",
            round(
                float(
                    item["rerank_score"]
                ),
                4
            )
        )

        print(
            "Source URL:",
            item["source_url"]
        )

        print("\nInformation:")

        print(
            item["text"]
        )

    # --------------------------------------------------------
    # LLM CONTEXT
    # --------------------------------------------------------

    print("\n===================================")
    print("CONTEXT FOR LLM")
    print("===================================")

    print(
        result.get(
            "context",
            ""
        )
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    from retriever import (
        load_vector_store,
        load_embedding_model,
        retrieve
    )

    from reranker import (
        load_reranker,
        rerank
    )

    print("===================================")
    print("TN FARMER SCHEME AI")
    print("RAG - GUARDRAILS")
    print("===================================")

    print("\n[1] Loading vector store...")

    index, chunks = load_vector_store()

    print(
        "FAISS vectors:",
        index.ntotal
    )

    print("\n[2] Loading embedding model...")

    embedding_model = load_embedding_model()

    # --------------------------------------------------------
    # TEST QUESTION
    # --------------------------------------------------------

    query = (
        "What subsidy is available for farmers "
        "for agricultural machinery?"
    )

    print("\n[3] Farmer Question:")

    print(query)

    # --------------------------------------------------------
    # QUESTION CHECK
    # --------------------------------------------------------

    print("\n[4] Checking question...")

    question_status = get_question_status(query)

    print(
        "Question status:",
        question_status["status"]
    )

    if question_status["message"]:

        print(
            "Message:",
            question_status["message"]
        )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    if question_status["status"] == "VALID":

        print(
            "\n[5] Retrieving candidates..."
        )

        retrieved_results = retrieve(
            query=query,
            index=index,
            chunks=chunks,
            model=embedding_model,
            top_k=5
        )

        print(
            "Retrieved candidates:",
            len(retrieved_results)
        )

        # ----------------------------------------------------
        # RERANKING
        # ----------------------------------------------------

        print(
            "\n[6] Reranking candidates..."
        )

        reranker_model = load_reranker()

        reranked_results = rerank(
            query=query,
            results=retrieved_results,
            model=reranker_model,
            top_k=3
        )

        print(
            "Reranked results:",
            len(reranked_results)
        )

        # ----------------------------------------------------
        # GUARDRAILS
        # ----------------------------------------------------

        print(
            "\n[7] Applying guardrails..."
        )

        guardrail_result = apply_guardrails(
            query=query,
            results=reranked_results
        )

        display_guardrail_result(
            guardrail_result
        )

    print("\n===================================")
    print("GUARDRAILS COMPLETED")
    print("===================================")