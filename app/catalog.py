from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without faiss
    faiss = None


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "assessment",
    "hiring",
    "be",
    "for",
    "from",
    "how",
    "i",
    "in",
    "interviewing",
    "is",
    "it",
    "concepts",
    "need",
    "of",
    "or",
    "work",
    "works",
    "role",
    "should",
    "the",
    "to",
    "what",
    "with",
    "we",
    "who",
}


TYPE_MAP = {
    "ability & aptitude": "A",
    "knowledge & skills": "K",
    "personality & behavior": "P",
    "simulations": "S",
    "biodata & situational judgment": "S",
    "competencies": "C",
    "development & 360": "D",
    "assessment exercises": "E",
}


@dataclass(frozen=True)
class CatalogItem:
    entity_id: str
    name: str
    link: str
    description: str
    job_levels: list[str]
    languages: list[str]
    duration: str
    status: str
    remote: str
    adaptive: str
    keys: list[str]

    @property
    def is_packaged_solution(self) -> bool:
        name = self.name.lower()
        return bool(re.search(r"\bsolution\b", name)) or bool(re.search(r"\bguide\b", name))

    @property
    def test_type(self) -> str:
        for key in self.keys:
            mapped = TYPE_MAP.get(key.lower())
            if mapped:
                return mapped
        return TYPE_MAP.get(self.keys[0].lower(), "U") if self.keys else "U"

    def search_text(self) -> str:
        parts = [
            self.name,
            self.description,
            " ".join(self.keys),
            " ".join(self.job_levels),
            " ".join(self.languages),
            self.duration,
        ]
        return " ".join(part for part in parts if part)


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def _catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "shl_product_catalog.json"


def _load_items() -> list[CatalogItem]:
    raw = json.loads(_catalog_path().read_text(encoding="utf-8"), strict=False)
    items: list[CatalogItem] = []
    for entry in raw:
        items.append(
            CatalogItem(
                entity_id=str(entry.get("entity_id", "")),
                name=str(entry.get("name", "")),
                link=str(entry.get("link", "")),
                description=str(entry.get("description", "")),
                job_levels=list(entry.get("job_levels", []) or []),
                languages=list(entry.get("languages", []) or []),
                duration=str(entry.get("duration", "")),
                status=str(entry.get("status", "")),
                remote=str(entry.get("remote", "")),
                adaptive=str(entry.get("adaptive", "")),
                keys=list(entry.get("keys", []) or []),
            )
        )
    return items


class CatalogIndex:
    def __init__(self) -> None:
        self.all_items = _load_items()
        self.items = [item for item in self.all_items if not item.is_packaged_solution]
        self._name_lookup = {self._normalize(item.name): item for item in self.items}
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._item_matrix: np.ndarray | None = None
        self._index = None
        self._build_index()

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _build_index(self) -> None:
        tokenized_docs = [_tokenize(item.search_text()) for item in self.items]
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_docs:
            doc_freq.update(set(tokens))

        vocab_items = [token for token, count in doc_freq.items() if count >= 2]
        vocab_items.sort(key=lambda token: (-doc_freq[token], token))
        if len(vocab_items) > 6000:
            vocab_items = vocab_items[:6000]
        self._vocab = {token: idx for idx, token in enumerate(vocab_items)}

        total_docs = max(len(self.items), 1)
        self._idf = np.array(
            [math.log((1.0 + total_docs) / (1.0 + doc_freq[token])) + 1.0 for token in vocab_items],
            dtype=np.float32,
        )

        matrix = np.zeros((len(self.items), len(self._vocab)), dtype=np.float32)
        for row, tokens in enumerate(tokenized_docs):
            counts = Counter(token for token in tokens if token in self._vocab)
            if not counts:
                continue
            total = float(sum(counts.values()))
            for token, count in counts.items():
                col = self._vocab[token]
                matrix[row, col] = (count / total) * self._idf[col]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._item_matrix = matrix

        if faiss is not None and matrix.size:
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._index = index

    def vectorize(self, text: str) -> np.ndarray:
        if not self._vocab:
            return np.zeros((1, 0), dtype=np.float32)
        vector = np.zeros((len(self._vocab),), dtype=np.float32)
        tokens = [token for token in _tokenize(text) if token in self._vocab]
        if not tokens:
            return vector.reshape(1, -1)
        counts = Counter(tokens)
        total = float(sum(counts.values()))
        for token, count in counts.items():
            col = self._vocab[token]
            vector[col] = (count / total) * self._idf[col]
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        return vector.reshape(1, -1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[CatalogItem, float]]:
        if not self.items:
            return []
        vector = self.vectorize(query)
        if vector.shape[1] == 0:
            return []

        if self._index is not None:
            scores, indices = self._index.search(vector, min(top_k, len(self.items)))
            results: list[tuple[CatalogItem, float]] = []
            for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
                if idx < 0:
                    continue
                results.append((self.items[idx], float(score)))
            return results

        assert self._item_matrix is not None
        scores = self._item_matrix @ vector[0]
        order = np.argsort(-scores)[:top_k]
        return [(self.items[int(idx)], float(scores[int(idx)])) for idx in order]

    def find_exact(self, query: str) -> CatalogItem | None:
        normalized = self._normalize(query)
        if not normalized:
            return None
        if normalized in self._name_lookup:
            return self._name_lookup[normalized]
        best_item = None
        best_score = 0.0
        query_tokens = set(_tokenize(query))
        for item in self.items:
            name_norm = self._normalize(item.name)
            if normalized in name_norm or name_norm in normalized:
                return item
            name_tokens = set(_tokenize(item.name))
            overlap = len(query_tokens & name_tokens)
            ratio = overlap / max(len(name_tokens), 1)
            if ratio > best_score:
                best_score = ratio
                best_item = item
        return best_item if best_score >= 0.25 else None

    def find_best(self, query: str) -> CatalogItem | None:
        exact = self.find_exact(query)
        if exact is not None:
            return exact
        results = self.search(query, top_k=1)
        if results and results[0][1] > 0:
            return results[0][0]
        return None


@lru_cache(maxsize=1)
def get_catalog() -> CatalogIndex:
    return CatalogIndex()


def map_test_type(item: CatalogItem) -> str:
    return item.test_type
