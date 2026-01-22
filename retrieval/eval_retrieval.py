from retrieval.retriever import Retriever

QUESTIONS = [
    "What are the production dos for RAG?",
    "What is the difference between standard retrieval and the ColPali approach?",
    "Why is hybrid search better than vector-only search?"
]


def run_eval():
    retriever = Retriever(top_k=5, rerank_k=3)

    for q in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"❓ QUESTION: {q}")
        print("=" * 80)

        docs = retriever.retrieve(q)

        for i, doc in enumerate(docs, 1):
            print(f"\n--- Retrieved Chunk {i} ---\n")
            print(doc[:800])  # enough to judge relevance


if __name__ == "__main__":
    run_eval()
