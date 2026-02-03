# 📘 RAG Chatbot (Local, CPU-Only)

A local Retrieval-Augmented Generation (RAG) chatbot that ingests PDFs and lecture videos, builds a searchable knowledge base using embeddings, and answers user questions grounded strictly in the ingested content.

This project is CPU-only, fully offline, and designed for correctness, reproducibility, and explainability, not real-time latency.


## 🔹 Key Features

📄 Multi-format ingestion: PDFs + lecture videos

🎧 Speech-to-text using Whisper

✂️ Semantic chunking with overlap

🧠 Vector search using ChromaDB

🤖 Local LLM inference (Ollama)

🔁 Idempotent pipeline (safe to rerun)

🪵 Structured logging

❌ No cloud APIs, no paid services

## 🧱 High-Level Architecture
```
PDFs / Videos
    ↓
Text Extraction + Transcription
    ↓
Cleaning & Merging
    ↓
Chunking
    ↓
Embedding (Sentence Transformers)
    ↓
ChromaDB (Vector Store)
    ↓
Local LLM (RAG-based Answering)
```
## 🛠️ Tech Stack

| Layer               | Technology                                 |
| ------------------- | ------------------------------------------ |
| Language            | Python 3.10                                |
| PDF Parsing         | PyMuPDF                                    |
| Audio Transcription | OpenAI Whisper (local)                     |
| Embeddings          | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB           | ChromaDB (persistent, local)               |
| LLM                 | Ollama (local models like Mistral / Phi-3) |
| Chunking            | Custom word-based chunking with overlap    |
| Logging             | Python logging                             |
| Platform            | CPU-only, Windows/Linux                    |

## 📁 Project Structure
```
RAG_chatbot/
├── data/
│   ├── pdfs/              # Input PDFs
│   ├── audio/             # Input videos (.mp4)
│   ├── audio_wav/         # Extracted audio
│   ├── transcripts/       # Transcripts + merged clean text
│   └── chunks/            # Text chunks
├── ingestion/             # PDF & audio ingestion
├── processing/            # Cleaning & chunking
├── embeddings/            # Embedding + ChromaDB logic
├── retrieval/             # Retrieval utilities
├── generation/            # RAG prompt + LLM logic
├── utils/                 # Logging & helpers
├── rag_pipeline.py        # End-to-end ingestion pipeline
├── app.py                 # Interactive CLI chatbot
├── requirements.txt
└── README.md
```

## 📥 How to Add New Data

1️⃣ Add PDFs

Place all PDF files into:
```
data/pdfs/
```
2️⃣ Add Videos

Place lecture videos (.mp4) into:
```
data/audio/
```
🔄 Run the RAG Pipeline

The pipeline performs extraction → cleaning → chunking → embedding → vector store rebuild.
```
python rag_pipeline.py
```

### What this does:

Extracts text from PDFs

Converts videos to audio and transcribes them

Cleans and merges all text

Deletes old chunks

Creates new chunks

Rebuilds ChromaDB embeddings from scratch

✅ Safe to run multiple times

✅ No duplication

✅ Deterministic behavior

## 💬 Run the Chatbot
Interaction is via a local CLI chatbot.
```
python app.py
```

Usage :
```
❓ Ask a question: Ask any question about Vector database
```
Type exit to quit.


## 🧠 How RAG Is Enforced

🧠 How RAG Is Enforced

Queries retrieve relevant chunks from ChromaDB

Retrieved context is injected into the prompt

The LLM is instructed to answer only using retrieved context

If context is insufficient → responds with “I don’t know”

This reduces hallucination and improves trustworthiness.

## ⏱️ Performance Notes

Retrieval latency: sub-second

End-to-end latency: higher due to CPU-only local LLM inference

This is an intentional trade-off

Performance can be improved with:

Smaller models

Model warm-up

GPU inference

##🔐 Design Decisions

Rebuild-based ingestion (no incremental updates)

Local-only execution

No background jobs

Focus on clarity and correctness

These choices make the system easy to reason about and evaluate.

## 🧪 Limitations

Not real-time

CPU-only

No API / UI

Not optimized for very large corpora

All limitations are intentional

## 🚀 Possible Extensions

Metadata-aware chunks (source, page, timestamp)

FastAPI wrapper

Dockerization

Retrieval quality evaluation metrics

## 👨‍💻 Author

Abdul Rahaman S | AI/ML Engineer