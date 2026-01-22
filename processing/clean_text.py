import re
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Light text cleaning suitable for RAG.
    Avoids aggressive spell correction.
    """

    # Remove multiple newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove common filler patterns
    fillers = [
        r"\buh\b", r"\bum\b", r"\byou know\b", r"\bkind of\b"
    ]

    for filler in fillers:
        text = re.sub(filler, "", text, flags=re.IGNORECASE)

    return text.strip()


def merge_and_clean(pdf_text_path: str, transcript_dir: str, output_file: str):
    pdf_text = Path(pdf_text_path).read_text(encoding="utf-8")

    transcripts = []
    for file in Path(transcript_dir).glob("*.txt"):
        transcripts.append(
            f"\n--- {file.name} ---\n" +
            file.read_text(encoding="utf-8")
        )

    merged_text = pdf_text + "\n".join(transcripts)
    cleaned_text = clean_text(merged_text)

    Path(output_file).write_text(cleaned_text, encoding="utf-8")
    print(f"✅ Clean merged text saved to {output_file}")


if __name__ == "__main__":
    merge_and_clean(
        pdf_text_path="data/transcripts/pdf_text.txt",
        transcript_dir="data/transcripts",
        output_file="data/transcripts/merged_clean_text.txt"
    )
