from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from retrieval.bm25_index import BM25Index
from utils.logger import logger



DB_DIR = "vector_db"
COLLECTION_NAME = "rag_knowledge"
CHUNK_DIR = "data/chunks"
MAX_CHUNK_CHARS = 800

_cached_embedder = None
_cached_bm25 = None
_cached_chunks = None



class HybridRetriever:
    def __init__(self, top_k: int = 5, final_k: int = 2):

        global _cached_embedder, _cached_bm25, _cached_chunks

        self.top_k = top_k
        self.final_k = final_k

        # Vector components
        self.client = PersistentClient(path=DB_DIR)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        if _cached_embedder is None:
            _cached_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedder = _cached_embedder

        # BM25 component
        if _cached_bm25 is None:
            _cached_bm25 = BM25Index(CHUNK_DIR)
        self.bm25 = _cached_bm25

        # Load chunks
        if _cached_chunks is None:
            _cached_chunks = {
                file.stem: file.read_text(encoding="utf-8")
                for file in Path(CHUNK_DIR).glob("*.txt")
        }
        self.chunk_texts = _cached_chunks    

    def retrieve(self, query: str):
        
    
        # ---------- Vector Retrieval ----------

        logger.info("Starting hybrid retrieval")
        
        query_embedding = self.embedder.encode(query)

        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.top_k
        )
        

        vector_docs = vector_results["documents"][0]
        vector_ids = vector_results["ids"][0]
        logger.info("Vector retrieval completed")

        # ---- Vector Re-ranking ----
        

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
        
        logger.info("BM25 retrieval completed")
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
        return [
                 self.chunk_texts[doc_id][:MAX_CHUNK_CHARS]
                    for doc_id in final_ids]
    logger.info("Hybrid ranking completed")

if __name__ == "__main__":
    retriever = HybridRetriever()

    queries = [
        "What are the production dos for RAG?",
            ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"❓ QUESTION: {q}")
        print("=" * 80)

        docs = retriever.retrieve(q)
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Chunk {i} ---\n")
            print(doc[:800])
