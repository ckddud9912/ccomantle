# ---- 복붙 시작 ----
import os
import gzip
import shutil
import requests
import numpy as np

# FastText 한국어 벡터 (Facebook 공식)
VECTOR_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.vec.gz"
VECTOR_GZ = "cc.ko.300.vec.gz"
VECTOR_FILE = "cc.ko.300.vec"

word_vectors: dict[str, np.ndarray] = {}
_loaded = False

_answer_cache = None
_sims_cache = None  # list[(word, raw_sim)]
_ranking_cache = None  # dict[word, rank]


def _download_vectors():
    print("📥 FastText 벡터 다운로드 시작...")
    resp = requests.get(VECTOR_URL, stream=True)
    resp.raise_for_status()
    with open(VECTOR_GZ, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print("📥 다운로드 완료:", VECTOR_GZ)


def _extract_vectors():
    print("🧩 FastText 벡터 압축 해제 중...")
    with gzip.open(VECTOR_GZ, "rb") as f_in, open(VECTOR_FILE, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print("🧩 압축 해제 완료:", VECTOR_FILE)
    try:
        os.remove(VECTOR_GZ)
    except OSError:
        pass


def _ensure_vectors_ready():
    if not os.path.exists(VECTOR_FILE):
        if not os.path.exists(VECTOR_GZ):
            _download_vectors()
        _extract_vectors()


def load_fasttext():
    """cc.ko.300.vec 전체를 메모리에 올림 (첫 호출만 오래 걸림)."""
    global _loaded, word_vectors
    if _loaded:
        return

    _ensure_vectors_ready()

    print("🔧 FastText 벡터 로딩 시작...")
    with open(VECTOR_FILE, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 302:
                continue
            w = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            word_vectors[w] = vec

    _loaded = True
    print("✅ FastText 로딩 완료. 단어 수:", len(word_vectors))


def has_word(word: str) -> bool:
    return word in word_vectors


def get_vector(word: str) -> np.ndarray | None:
    return word_vectors.get(word)


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return dot / norm if norm > 0 else 0.0


def convert_similarity(sim: float) -> float:
    """
    Raw cosine 유사도(sim, 대략 0.2~0.8)를
    꼬맨틀 느낌으로 -20 ~ +70 점수로 변환
    """
    scaled = ((sim - 0.2) / 0.6) * 90.0 - 20.0
    if scaled < -20:
        scaled = -20
    if scaled > 70:
        scaled = 70
    return round(float(scaled), 2)


def calculate_ranking(answer_word: str):
    """
    정답 기준 전체 단어 유사도 순위 계산.
    """
    global _answer_cache, _sims_cache, _ranking_cache

    load_fasttext()

    if _answer_cache == answer_word and _sims_cache is not None:
        return _sims_cache, _ranking_cache

    answer_vec = word_vectors[answer_word]
    sims = []

    print("📊 전체 단어 유사도 계산 중... (시간 소요)")
    for w, vec in word_vectors.items():
        s = cosine_sim(vec, answer_vec)
        sims.append((w, s))

    sims.sort(key=lambda x: x[1], reverse=True)
    ranking = {w: idx + 1 for idx, (w, _) in enumerate(sims)}

    _answer_cache = answer_word
    _sims_cache = sims
    _ranking_cache = ranking

    print("🏁 순위 테이블 생성 완료!")
    return sims, ranking


def stats_for_answer(answer_word: str):
    """
    정답 기준:
      - 가장 유사한 단어
      - 10번째 유사한 단어
      - 1000번째 유사한 단어
    환산 점수 리턴
    """
    sims, _ = calculate_ranking(answer_word)
    filtered = [item for item in sims if item[0] != answer_word]

    def get_k(k: int):
        if len(filtered) >= k:
            return convert_similarity(filtered[k - 1][1])
        return None

    max_sim = get_k(1)
    top10_sim = get_k(10)
    top1000_sim = get_k(1000)
    return max_sim, top10_sim, top1000_sim
# ---- 복붙 끝 ----
