from generation.rag_chain import ask_rag

questions = [
    "What are the production dos for RAG?",
    "What is the difference between standard retrieval and the ColPali approach?",
    "Why is hybrid search better than vector-only search?"
]

with open("logs/test_answers.txt", "w", encoding="utf-8") as f:
    for q in questions:
        print("\n" + "=" * 80)
        print(f"❓ {q}")
        print("=" * 80)

        answer = ask_rag(q)
        print(answer)

        f.write(f"Question: {q}\n")
        f.write(f"Answer:\n{answer}\n")
        f.write("-" * 80 + "\n")
