from pathlib import Path
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient


CHUNK_DIR = "data/chunks"
DB_DIR = "vector_db"
COLLECTION_NAME = "rag_knowledge"


def load_chunks(chunk_dir: str):
    chunks = []
    for file in Path(chunk_dir).glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        chunks.append(
            {
                "id": file.stem,
                "text": text
            }
        )
    return chunks


def create_vector_store(chunks):
    client = PersistentClient(path=DB_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        ids=ids
    )

    print(f"✅ Stored {len(texts)} chunks in ChromaDB (persisted on disk)")


if __name__ == "__main__":
    chunks = load_chunks(CHUNK_DIR)
    create_vector_store(chunks)
