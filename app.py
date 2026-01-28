from generation.rag_chain import ask_rag

questions = [
    "What are the production dos for RAG?",
    "Tell me about Colpali approach"
    
    ]

for q in questions:
    print("\n" + "=" * 80)
    print(f"❓ {q}")
    print("=" * 80)

    answer = ask_rag(q)

    print("\n✅ ANSWER:\n")
    print(answer)

