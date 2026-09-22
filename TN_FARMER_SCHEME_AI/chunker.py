from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CHUNK SETTINGS
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ============================================================
# CREATE RAG DOCUMENT TEXT
# ============================================================

def prepare_scheme_text(record):
    """
    Convert one structured scheme record into
    meaningful text for RAG.
    """

    return f"""
Department:
{record.get("department", "")}

Scheme Name:
{record.get("scheme_name", "")}

Description:
{record.get("description", "")}

Eligibility:
{record.get("eligibility", "")}

Documents Required:
{record.get("documents", "")}

Guidelines:
{record.get("guidelines_url", "")}

Source:
{record.get("source_url", "")}
""".strip()


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(records):
    """
    Convert scheme records into smaller RAG chunks.

    Returns:
        list of dictionaries containing:
        - text
        - metadata
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for record in records:

        scheme_text = prepare_scheme_text(record)

        split_texts = splitter.split_text(
            scheme_text
        )

        for chunk_number, text in enumerate(
            split_texts,
            start=1
        ):

            chunks.append({
                "text": text,
                "metadata": {
                    "id": record.get("id"),
                    "department": record.get("department"),
                    "scheme_name": record.get("scheme_name"),
                    "source_url": record.get("source_url"),
                    "chunk_number": chunk_number
                }
            })

    return chunks


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from loader import load_structured_data

    print("===================================")
    print("TN FARMER SCHEME AI")
    print("RAG - CHUNKER")
    print("===================================")

    # --------------------------------------------------------
    # Load structured records
    # --------------------------------------------------------

    records = load_structured_data()

    print("\nRecords received:", len(records))

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(records)

    print("Chunks created  :", len(chunks))

    # --------------------------------------------------------
    # Show first 3 chunks
    # --------------------------------------------------------

    print("\n===================================")
    print("SAMPLE CHUNKS")
    print("===================================")

    for index, chunk in enumerate(
        chunks[:3],
        start=1
    ):

        print(f"\nCHUNK {index}")
        print("-----------------------------------")

        print(
            "Scheme ID:",
            chunk["metadata"]["id"]
        )

        print(
            "Scheme:",
            chunk["metadata"]["scheme_name"]
        )

        print(
            "Chunk:",
            chunk["metadata"]["chunk_number"]
        )

        print("\nText:")
        print(chunk["text"])

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    print("\n===================================")
    print("CHUNKING COMPLETED")
    print("===================================")