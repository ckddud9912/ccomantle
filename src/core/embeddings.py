import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import orjson


@dataclass
class EmbeddingStore:
    words: List[str]
    matrix: np.ndarray  # (N, D), L2-normalized, float32
    word_to_idx: Dict[str, int]

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_idx

    def __len__(self) -> int:
        return len(self.words)

    def vector(self, word: str) -> np.ndarray:
        return self.matrix[self.word_to_idx[word]]


def load_store(path: str) -> EmbeddingStore:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    with open(path, "rb") as f:
        raw: Dict[str, List[float]] = orjson.loads(f.read())

    words = list(raw.keys())
    matrix = np.array([raw[w] for w in words], dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    word_to_idx = {w: i for i, w in enumerate(words)}
    return EmbeddingStore(words=words, matrix=matrix, word_to_idx=word_to_idx)
