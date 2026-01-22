import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text from a single PDF file.
    """
    doc = fitz.open(pdf_path)
    text = []

    for page in doc:
        page_text = page.get_text()
        if page_text:
            text.append(page_text)

    return "\n".join(text)


def extract_all_pdfs(pdf_dir: str, output_file: str):
    """
    Extracts text from all PDFs in a directory and saves to a single file.
    """
    pdf_dir = Path(pdf_dir)
    all_text = []

    for pdf_file in tqdm(list(pdf_dir.glob("*.pdf")), desc="Processing PDFs"):
        print(f"\nExtracting: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        all_text.append(f"\n--- {pdf_file.name} ---\n")
        all_text.append(text)

    Path(output_file).write_text("\n".join(all_text), encoding="utf-8")
    print(f"\n✅ PDF text saved to {output_file}")


if __name__ == "__main__":
    extract_all_pdfs(
        pdf_dir="data/pdfs",
        output_file="data/transcripts/pdf_text.txt"
    )
