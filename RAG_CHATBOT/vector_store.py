import chromadb


DB_PATH = "./chroma_db"
COLLECTION_NAME = "Parenting_Policy_Collection"

def store_in_chroma(chunks, embeddings):
    """
    Store text chunks and their embeddings in ChromaDB.
    """

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    print("Data stored successfully in ChromaDB!")
    print(f"Total chunks stored: {len(chunks)}")


def get_collection():
    """
    Get the collection.
    """

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection
def verify_storage():
    """
    Verify that chunks and embeddings are stored.
    """

    collection = get_collection()

    print("\n--- ChromaDB Verification ---")
    print(f"Total records: {collection.count()}")

    result = collection.get(
        ids=["chunk_0"],
        include=["documents", "embeddings"]
    )

    print("\nFirst chunk:")
    print(result["documents"][0][:500])

    print("\nFirst vector:")
    print(result["embeddings"][0][:10])

    print(f"\nVector dimensions: {len(result['embeddings'][0])}")