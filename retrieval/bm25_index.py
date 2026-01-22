from pathlib import Path
from rank_bm25 import BM25Okapi
import re


class BM25Index:
    def __init__(self, chunk_dir: str):
        self.documents = []
        self.doc_ids = []

        for file in Path(chunk_dir).glob("*.txt"):
            text = file.read_text(encoding="utf-8")
            tokens = self._tokenize(text)
            self.documents.append(tokens)
            self.doc_ids.append(file.stem)

        self.bm25 = BM25Okapi(self.documents)

    def _tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def query(self, query: str, top_k: int = 5):
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.doc_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc_id for doc_id, _ in ranked[:top_k]]
