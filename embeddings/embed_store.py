from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

DB_DIR = "vector_db"
COLLECTION_NAME = "rag_knowledge"


def create_vector_store(chunks):
    """
    chunks: List[str]
    """
    client = PersistentClient(path=DB_DIR)

    # 🔥 Rebuild strategy: delete existing collection
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ Deleted existing collection '{COLLECTION_NAME}'")

    collection = client.create_collection(name=COLLECTION_NAME)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = chunks
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    embeddings = model.encode(texts, show_progress_bar=True)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids
    )

    print(f"✅ Stored {len(texts)} chunks in ChromaDB (fresh rebuild)")
