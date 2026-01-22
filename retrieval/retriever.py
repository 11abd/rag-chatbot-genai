from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from retrieval.bm25_index import BM25Index


DB_DIR = "vector_db"
COLLECTION_NAME = "rag_knowledge"
CHUNK_DIR = "data/chunks"


class HybridRetriever:
    def __init__(self, top_k: int = 5, final_k: int = 3):
        self.top_k = top_k
        self.final_k = final_k

        # Vector components
        self.client = PersistentClient(path=DB_DIR)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # BM25 component
        self.bm25 = BM25Index(CHUNK_DIR)

        # Load chunks
        self.chunk_texts = {
            file.stem: file.read_text(encoding="utf-8")
            for file in Path(CHUNK_DIR).glob("*.txt")
        }

    def retrieve(self, query: str):
        # ---------- Vector Retrieval ----------
        query_embedding = self.embedder.encode(query)

        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.top_k
        )

        vector_docs = vector_results["documents"][0]
        vector_ids = vector_results["ids"][0]

        vector_embeddings = self.embedder.encode(vector_docs)
        vector_scores = cosine_similarity(
            [query_embedding],
            vector_embeddings
        )[0]

        vector_score_map = {
            doc_id: score
            for doc_id, score in zip(vector_ids, vector_scores)
        }

        # ---------- BM25 Retrieval ----------
        bm25_ids = self.bm25.query(query, top_k=self.top_k)
        bm25_score_map = {doc_id: 1.0 for doc_id in bm25_ids}

        # ---------- Score Fusion ----------
        combined_scores = {}

        for doc_id in set(vector_score_map) | set(bm25_score_map):
            combined_scores[doc_id] = (
                0.7 * vector_score_map.get(doc_id, 0.0) +
                0.3 * bm25_score_map.get(doc_id, 0.0)
            )

        ranked = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_ids = [doc_id for doc_id, _ in ranked[:self.final_k]]
        return [self.chunk_texts[doc_id] for doc_id in final_ids]


if __name__ == "__main__":
    retriever = HybridRetriever()

    queries = [
        "What are the production dos for RAG?",
        "What is the difference between standard retrieval and the ColPali approach?",
        "Why is hybrid search better than vector-only search?"
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"❓ QUESTION: {q}")
        print("=" * 80)

        docs = retriever.retrieve(q)
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Chunk {i} ---\n")
            print(doc[:800])
