from generation.rag_chain import ask_rag
from utils.logger import logger

def main():
    print("🔍 RAG Chatbot (type 'exit' to quit)\n")

    while True:
        question = input("❓ Ask a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("👋 Exiting RAG chatbot.")
            logger.info("User exited application")
            break

        logger.info(f"User question: {question}")

        try:
            answer = ask_rag(question)
            print("\n✅ Answer:\n")
            print(answer)
            print("\n" + "-" * 80 + "\n")

            logger.info("Answer successfully generated")

        except Exception as e:
            logger.exception("Error during RAG execution")
            print("❌ Error occurred. Check logs/app.log")

if __name__ == "__main__":
    main()

