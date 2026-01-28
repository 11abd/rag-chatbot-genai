from retrieval.retriever import HybridRetriever
from generation.llm import generate_answer
from utils.logger import logger




def build_prompt(question: str, context_chunks: list[str]) -> str:
    MAX_CHARS = 1800  # ~400–500 tokens total

    trimmed_chunks = []
    current_len = 0

    for chunk in context_chunks:
        chunk = chunk.strip()
        if current_len + len(chunk) > MAX_CHARS:
            remaining = MAX_CHARS - current_len
            if remaining > 0:
                trimmed_chunks.append(chunk[:remaining])
            break
        trimmed_chunks.append(chunk)
        current_len += len(chunk)

    context = "\n\n".join(trimmed_chunks)

    prompt = f"""
You are an expert assistant.
Answer the question ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt.strip()


retriever = HybridRetriever(top_k=5, final_k=2)
def ask_rag(question: str) -> str:
    logger.info("RAG query received")

    context_chunks = retriever.retrieve(question)
    logger.info("Context retrieved")

   
    prompt = build_prompt(question, context_chunks)
    logger.info("Prompt constructed")

 
    answer = generate_answer(prompt)
    logger.info("Answer generated")

    
    return answer
