import requests
import json
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_FILE = BASE_DIR / "raw_data.json"
STRUCTURED_DATA_FILE = BASE_DIR / "structured_data.json"


# ============================================================
# SOURCE
# ============================================================

SOURCE_URL = "https://www.tnagrisnet.tn.gov.in/people_app/goScheme/"

AJAX_URL = (
    "https://www.tnagrisnet.tn.gov.in/"
    "people_app/goScheme/pending_ajax_list"
)


# ============================================================
# STAGE 1: WEB INGESTION
# ============================================================

def fetch_scheme_data():
    """
    Fetch raw scheme records from the official
    TN Agrisnet portal.
    """

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": SOURCE_URL,
    }

    response = requests.get(
        AJAX_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data.get("data", [])


# ============================================================
# SAVE RAW DATA
# ============================================================

def save_raw_data(records):
    """
    Save the original raw records before cleaning.

    Important:
    Raw government data is preserved exactly as received.
    """

    raw_data = {
        "source_url": SOURCE_URL,
        "retrieved_date": datetime.now().isoformat(),
        "total_records": len(records),
        "records": records
    }

    with open(
        RAW_DATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            raw_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nRaw data saved to:")
    print(RAW_DATA_FILE)


# ============================================================
# STAGE 2: CLEANING
# ============================================================

def clean_text(value):
    """
    Remove HTML tags and unnecessary whitespace.
    """

    if value is None:
        return ""

    value = str(value)

    soup = BeautifulSoup(
        value,
        "html.parser"
    )

    value = soup.get_text(
        " ",
        strip=True
    )

    value = " ".join(value.split())

    return value.strip()


# ============================================================
# TEST / JUNK RECORD DETECTION
# ============================================================

def is_test_or_junk_record(
    scheme_name,
    description,
    eligibility,
    documents
):
    """
    Detect obvious test/junk records.

    The TN Agrisnet data currently contains records such as:

        tester '
        tester "
        tester `
        tester

    Therefore, checking only for exact equality with
    'tester' is not sufficient.

    This function intentionally uses a conservative rule:
    remove records containing the standalone word 'tester'
    in the main scheme fields.
    """

    fields = [
        scheme_name,
        description,
        eligibility,
        documents
    ]

    for field in fields:

        if not field:
            continue

        text = field.lower().strip()

        words = text.split()

        if "tester" in words:
            return True

        # Also handle values such as:
        # tester'
        # tester"
        # tester`
        # tester,
        # tester.
        cleaned = (
            text
            .replace("'", " ")
            .replace('"', " ")
            .replace("`", " ")
            .replace(",", " ")
            .replace(".", " ")
            .replace(";", " ")
            .replace(":", " ")
            .replace("-", " ")
            .split()
        )

        if "tester" in cleaned:
            return True

    return False


# ============================================================
# CLEAN RECORDS
# ============================================================

def clean_records(records):
    """
    Clean raw scheme records.

    Returns:
        cleaned_records
        removed_records
    """

    cleaned_records = []
    removed_records = []

    for record in records:

        # ----------------------------------------------------
        # Validate record structure
        # ----------------------------------------------------

        if len(record) < 7:

            removed_records.append({
                "id": (
                    record[0]
                    if len(record) > 0
                    else "Unknown"
                ),
                "reason": (
                    f"Invalid record structure: "
                    f"only {len(record)} fields"
                ),
                "raw_record": record
            })

            continue

        # ----------------------------------------------------
        # Extract fields
        # ----------------------------------------------------

        scheme_id = record[0]

        department = clean_text(
            record[1]
        )

        scheme_name = clean_text(
            record[2]
        )

        description = clean_text(
            record[3]
        )

        eligibility = clean_text(
            record[4]
        )

        documents = clean_text(
            record[5]
        )

        guidelines = clean_text(
            record[6]
        )

        # ----------------------------------------------------
        # Missing scheme name
        # ----------------------------------------------------

        if not scheme_name:

            removed_records.append({
                "id": scheme_id,
                "reason": "Missing scheme name",
                "raw_record": record
            })

            continue

        # ----------------------------------------------------
        # Test / junk record
        # ----------------------------------------------------

        if is_test_or_junk_record(
            scheme_name=scheme_name,
            description=description,
            eligibility=eligibility,
            documents=documents
        ):

            removed_records.append({
                "id": scheme_id,
                "reason": (
                    "Test/junk record detected "
                    "because 'tester' was found "
                    "in scheme data"
                ),
                "raw_record": record
            })

            continue

        # ----------------------------------------------------
        # Clean record
        # ----------------------------------------------------

        cleaned_record = {
            "id": scheme_id,
            "department": department,
            "scheme_name": scheme_name,
            "scheme_description": description,
            "eligibility": eligibility,
            "documents": documents,
            "guidelines_url": guidelines,
            "source_url": SOURCE_URL
        }

        cleaned_records.append(
            cleaned_record
        )

    return cleaned_records, removed_records


# ============================================================
# STAGE 3: STRUCTURE DETECTION
# ============================================================

def detect_structure(cleaned_records):
    """
    Convert cleaned records into a consistent
    structured format.
    """

    structured_records = []

    for record in cleaned_records:

        structured_record = {
            "id": record["id"],

            "department": record["department"],

            "scheme_name": record["scheme_name"],

            "description": record["scheme_description"],

            "eligibility": record["eligibility"],

            "documents": record["documents"],

            "guidelines_url": record["guidelines_url"],

            "source_url": record["source_url"]
        }

        structured_records.append(
            structured_record
        )

    return structured_records


# ============================================================
# SAVE STRUCTURED DATA
# ============================================================

def save_structured_data(structured_records):
    """
    Save the output of the ingestion pipeline.

    This file will later become the input
    for the RAG pipeline.
    """

    structured_data = {
        "source_url": SOURCE_URL,
        "retrieved_date": datetime.now().isoformat(),
        "total_records": len(structured_records),
        "records": structured_records
    }

    with open(
        STRUCTURED_DATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            structured_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nStructured data saved to:")
    print(STRUCTURED_DATA_FILE)


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    print("===================================")
    print("TN FARMER SCHEME AI")
    print("INGESTION PIPELINE")
    print("===================================")

    # ========================================================
    # STAGE 1: WEB INGESTION
    # ========================================================

    print("\n[STAGE 1] WEB INGESTION")

    raw_records = fetch_scheme_data()

    print("Source URL  :", SOURCE_URL)
    print("Raw records :", len(raw_records))

    # --------------------------------------------------------
    # Save original raw data
    # --------------------------------------------------------

    save_raw_data(
        raw_records
    )

    # ========================================================
    # STAGE 2: CLEANING
    # ========================================================

    print("\n[STAGE 2] CLEANING")

    cleaned_records, removed_records = clean_records(
        raw_records
    )

    print("Clean records  :", len(cleaned_records))
    print("Removed records:", len(removed_records))

    # --------------------------------------------------------
    # Show removed records
    # --------------------------------------------------------

    print("\n===================================")
    print("REMOVED RECORDS")
    print("===================================")

    if not removed_records:

        print("No records were removed.")

    else:

        for i, removed in enumerate(
            removed_records,
            start=1
        ):

            print(f"\nREMOVED RECORD {i}")
            print("-----------------------------------")

            print("ID     :", removed["id"])
            print("Reason :", removed["reason"])
            print("Raw    :", removed["raw_record"])

    # ========================================================
    # STAGE 3: STRUCTURE DETECTION
    # ========================================================

    print("\n[STAGE 3] STRUCTURE DETECTION")

    structured_records = detect_structure(
        cleaned_records
    )

    print(
        "Structured records:",
        len(structured_records)
    )

    # --------------------------------------------------------
    # Save structured data
    # --------------------------------------------------------

    save_structured_data(
        structured_records
    )

    # --------------------------------------------------------
    # Show first 3 structured records
    # --------------------------------------------------------

    print("\n===================================")
    print("FIRST 3 STRUCTURED RECORDS")
    print("===================================")

    for i, record in enumerate(
        structured_records[:3],
        start=1
    ):

        print(f"\nRECORD {i}")
        print("-----------------------------------")

        print("ID          :", record["id"])
        print("Department  :", record["department"])
        print("Scheme Name :", record["scheme_name"])
        print("Description :", record["description"])
        print("Eligibility :", record["eligibility"])
        print("Documents   :", record["documents"])
        print("Guidelines  :", record["guidelines_url"])

        print("-----------------------------------")

    # ========================================================
    # FINAL INGESTION SUMMARY
    # ========================================================

    print("\n===================================")
    print("INGESTION SUMMARY")
    print("===================================")

    print("Raw records        :", len(raw_records))
    print("Clean records      :", len(cleaned_records))
    print("Removed records    :", len(removed_records))
    print("Structured records :", len(structured_records))

    print("\nFiles created:")
    print("1.", RAW_DATA_FILE)
    print("2.", STRUCTURED_DATA_FILE)

    print("\n===================================")
    print("INGESTION COMPLETED")
    print("===================================")