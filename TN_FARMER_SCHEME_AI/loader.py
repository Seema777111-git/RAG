import json
from pathlib import Path


# ============================================================
# FILE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STRUCTURED_DATA_FILE = BASE_DIR / "structured_data.json"


# ============================================================
# LOAD STRUCTURED DATA
# ============================================================

def load_structured_data():
    """
    Load structured scheme data created by ingestion.py.

    Returns:
        list: Scheme records
    """

    # --------------------------------------------------------
    # Check file exists
    # --------------------------------------------------------

    if not STRUCTURED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"structured_data.json not found at:\n"
            f"{STRUCTURED_DATA_FILE}"
        )

    # --------------------------------------------------------
    # Read JSON
    # --------------------------------------------------------

    with open(
        STRUCTURED_DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # --------------------------------------------------------
    # Validate JSON
    # --------------------------------------------------------

    if not isinstance(data, dict):
        raise ValueError(
            "structured_data.json must contain a JSON object."
        )

    if "records" not in data:
        raise ValueError(
            "The JSON file does not contain 'records'."
        )

    records = data["records"]

    if not isinstance(records, list):
        raise ValueError(
            "'records' must be a list."
        )

    return records


# ============================================================
# TEST LOADER
# ============================================================

def main():

    print("===================================")
    print("TN FARMER SCHEME AI")
    print("RAG - LOADER")
    print("===================================")

    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    records = load_structured_data()

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n[LOADER]")
    print("-----------------------------------")

    print("Source file:")
    print(STRUCTURED_DATA_FILE)

    print("\nRecords loaded:")
    print(len(records))

    # --------------------------------------------------------
    # Show first 3 records
    # --------------------------------------------------------

    print("\n===================================")
    print("FIRST 3 RECORDS")
    print("===================================")

    for index, record in enumerate(
        records[:3],
        start=1
    ):

        print(f"\nRECORD {index}")
        print("-----------------------------------")

        print("ID          :", record.get("id"))
        print(
            "Department  :",
            record.get("department")
        )
        print(
            "Scheme Name :",
            record.get("scheme_name")
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n===================================")
    print("LOADER TEST COMPLETED")
    print("===================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()