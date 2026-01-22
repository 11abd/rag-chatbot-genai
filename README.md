# RAG Chatbot – PDFs + Lecture Videos

An end-to-end Retrieval-Augmented Generation (RAG) chatbot built from scratch using
PDF lecture slides and recorded lecture videos as a multi-modal knowledge base.

The system supports document-grounded question answering using hybrid retrieval
(vector + keyword search) and a local LLM, with no paid APIs.

---

## 🚀 Features

- Multi-modal ingestion (PDFs + lecture videos)
- Audio transcription using Whisper (CPU-only)
- Semantic chunking and embedding
- Persistent vector storage using ChromaDB
- Hybrid retrieval (Vector + BM25)
- Grounded answer generation using a local LLM (Ollama + Mistral)
- Fully reproducible, local-first pipeline

---

## 🧠 Architecture

PDFs / Videos
↓
Text Extraction & Transcription
↓
Cleaning & Chunking
↓
Embeddings (Sentence Transformers)
↓
Vector DB (ChromaDB)
↓
Hybrid Retrieval (Vector + BM25)
↓
LLM Generation (Ollama)

---

## 🛠️ Tech Stack

- Python
- Whisper (open-source)
- Sentence Transformers
- ChromaDB
- BM25 (rank-bm25)
- Ollama (Mistral)
- VS Code, Windows CMD

---

## ▶️ How to Run

### 1. Create environment
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### 2. Install Ollama & model
ollama pull mistral

###3. Run the chatbot
python app.py

Answers will be printed in the terminal and saved to:

logs/test_answers.txt

---

📊 Evaluation

The system was evaluated using predefined lecture questions.
Retrieval quality was assessed via human-in-the-loop analysis
(Recall@k and ranking relevance).

Hybrid retrieval significantly improved recall for keyword-heavy
queries, while sparse concepts were identified as known limitations
without metadata enrichment.


🔍 Example Questions

What are the production do’s for RAG?

What is the difference between standard retrieval and the ColPali approach?

Why is hybrid search better than vector-only search?


📌 Notes

Runs fully locally (CPU-only)

No paid APIs required

Vector DB can be rebuilt at any time

Easily extensible with new PDFs or videos