from utils.logger import logger

from ingestion.load_pdf import extract_all_pdfs
from ingestion.extract_audio import process_all_videos
from ingestion.transcribe_audio import transcribe_all

from processing.clean_text import merge_and_clean
from processing.chunk_text import chunk_text, clear_old_chunks,save_chunks

from embeddings.embed_store import create_vector_store
from pathlib import Path
from typing import List



def run_rag_pipeline():
    logger.info("RAG PIPELINE STARTED")

    # 1️⃣ PDF extraction
    extract_all_pdfs(
        pdf_dir="data/pdfs",
        output_file="data/transcripts/pdf_text.txt"
    )
    logger.info("PDF extraction completed")

    # 2️⃣ Audio → WAV
    process_all_videos(
        video_dir="data/audio",
        output_dir="data/audio_wav"
    )
    logger.info("Audio extraction completed")

    # 3️⃣ Transcription
    transcribe_all(
        audio_dir="data/audio_wav",
        output_dir="data/transcripts"
    )
    logger.info("Audio transcription completed")

    # 4️⃣ Clean + merge
    merge_and_clean(
        pdf_text_path="data/transcripts/pdf_text.txt",
        transcript_dir="data/transcripts",
        output_file="data/transcripts/merged_clean_text.txt"
    )
    logger.info("Text cleaning & merging completed")

    # 5️⃣ Chunking
    merged_text = Path(
        "data/transcripts/merged_clean_text.txt"
    ).read_text(encoding="utf-8")

    clear_old_chunks("data/chunks")

    chunks = chunk_text(
        text=merged_text,
        max_words=250,
        overlap=50
    )
    save_chunks(chunks, "data/chunks")
    logger.info(f"Chunking completed | chunks={len(chunks)}")

    # 6️⃣ Embedding + Chroma

    create_vector_store(chunks)
    logger.info("Embeddings stored in ChromaDB")

    logger.info("RAG PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_rag_pipeline()

