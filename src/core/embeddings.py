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
    """파일 확장자에 따라 분기. .json (text) / .npz (binary, ~10x smaller)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    if path.endswith(".npz"):
        return _load_npz(path)
    if path.endswith(".json"):
        return _load_json(path)
    raise ValueError(f"지원 안 함: {path}. .json 또는 .npz 만 지원")


def _load_json(path: str) -> EmbeddingStore:
    with open(path, "rb") as f:
        raw: Dict[str, List[float]] = orjson.loads(f.read())

    words = list(raw.keys())
    matrix = np.array([raw[w] for w in words], dtype=np.float32)

    return _normalize_and_pack(words, matrix)


def _load_npz(path: str) -> EmbeddingStore:
    """convert_to_npz.py 가 생성한 단일 .npz 파일 로드.

    Expects keys:
        words   — object array of str (단어 순서 = matrix row 순서)
        vectors — float32 (N, D) matrix
    """
    data = np.load(path, allow_pickle=True)
    if "words" not in data or "vectors" not in data:
        raise ValueError(f"npz 에 'words'/'vectors' 키 없음: {path}")
    words = data["words"].tolist()
    matrix = np.array(data["vectors"], dtype=np.float32)
    return _normalize_and_pack(words, matrix)


def _normalize_and_pack(words: List[str], matrix: np.ndarray) -> EmbeddingStore:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    word_to_idx = {w: i for i, w in enumerate(words)}
    return EmbeddingStore(words=words, matrix=matrix, word_to_idx=word_to_idx)
