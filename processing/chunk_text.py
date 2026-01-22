from pathlib import Path
from typing import List
import shutil

def clear_old_chunks(output_dir: str):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)


def chunk_text(
    text: str,
    max_words: int = 250,
    overlap: int = 50
) -> List[str]:
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap

    return chunks


def save_chunks(chunks: List[str], output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):
        (output_dir / f"chunk_{i}.txt").write_text(chunk, encoding="utf-8")

    print(f"✅ Saved {len(chunks)} chunks to {output_dir}")


if __name__ == "__main__":
    text = Path("data/transcripts/merged_clean_text.txt").read_text(encoding="utf-8")

    clear_old_chunks("data/chunks")

    chunks = chunk_text(text)
    save_chunks(chunks, "data/chunks")

