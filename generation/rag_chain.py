from retrieval.retriever import HybridRetriever
from generation.llm import generate_answer


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an expert assistant.
Answer the question ONLY using the context below.
If the answer is not contained in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt.strip()


def ask_rag(question: str) -> str:
    retriever = HybridRetriever(top_k=5, final_k=3)
    context_chunks = retriever.retrieve(question)
    prompt = build_prompt(question, context_chunks)
    return generate_answer(prompt)


if __name__ == "__main__":
    q = "What are the production dos for RAG?"
    print(ask_rag(q))
