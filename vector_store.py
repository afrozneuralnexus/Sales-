"""
VectorStore – lightweight TF-IDF based retrieval.
No external vector database required; works fully offline.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.document_processor import DocumentChunk


class VectorStore:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self._matrix = None

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self._matrix = self.vectorizer.fit_transform(texts)

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if self._matrix is None or not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0.0]

    def __len__(self) -> int:
        return len(self.chunks)
