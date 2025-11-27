import json
import os
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import numpy as np


# ========= 설정값 =========
WORDS_FILE = "words_5000.json"           # 입력: 상위 5000 단어 리스트
OUTPUT_FILE = "embedding_dictionary.json"  # 출력: 단어 -> 임베딩 벡터
MODEL_NAME = "jhgan/ko-sroberta-multitask"  # Sentence-BERT 한국어 멀티태스크 모델
BATCH_SIZE = 64
# =========================


def load_words(path: str) -> List[str]:
    """words_5000.json 에서 단어 리스트 로드"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 파일을 찾을 수 없습니다. "
                                f"프로젝트 루트에 {path} 가 있는지 확인하세요.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} 내용 형식이 리스트가 아닙니다. "
                         f'형식은 ["단어1", "단어2", ...] 이어야 합니다.')

    # 문자열만 필터링 + 중복 제거
    words = [w for w in data if isinstance(w, str)]
    seen = set()
    unique_words = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)

    print(f"[정보] 총 단어 수: {len(words)}개, 중복 제거 후: {len(unique_words)}개")
    return unique_words


def load_model(model_name: str) -> SentenceTransformer:
    """Sentence-BERT 모델 로드"""
    print(f"[정보] 모델 로드 중: {model_name}")
    model = SentenceTransformer(model_name)
    print("[정보] 모델 로드 완료")
    return model


def compute_embeddings(
    model: SentenceTransformer,
    words: List[str],
    batch_size: int = 64
) -> Dict[str, List[float]]:
    """
    단어 리스트를 받아 임베딩 계산 후
    word -> embedding(list[float]) 형태 dict 로 반환
    """
    print("[정보] 임베딩 계산 시작")

    embeddings = model.encode(
        words,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    if embeddings.shape[0] != len(words):
        raise RuntimeError(
            f"임베딩 개수({embeddings.shape[0]})와 단어 개수({len(words)})가 다릅니다."
        )

    dim = embeddings.shape[1]
    print(f"[정보] 임베딩 계산 완료 - 개수: {len(words)}개, 차원: {dim}차원")

    embedding_dict: Dict[str, List[float]] = {}
    for word, vec in zip(words, embeddings):
        embedding_dict[word] = vec.tolist()

    return embedding_dict


def save_embedding_dict(path: str, embedding_dict: Dict[str, List[float]]) -> None:
    """임베딩 사전을 JSON 파일로 저장"""
    if os.path.exists(path):
        print(f"[경고] {path} 가 이미 존재합니다. 기존 파일을 덮어씁니다.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(embedding_dict, f, ensure_ascii=False)

    print(f"[완료] 임베딩 사전 저장 완료: {path}")
    print(f"[정보] 총 단어 수: {len(embedding_dict)}개")


def main():
    print("=== embedding_precompute.py 실행 시작 ===")

    # 1) 단어 로드
    words = load_words(WORDS_FILE)

    if not words:
        raise RuntimeError("단어 리스트가 비어 있습니다. words_5000.json 내용을 확인하세요.")

    # 2) 모델 로드
    model = load_model(MODEL_NAME)

    # 3) 임베딩 계산
    embedding_dict = compute_embeddings(model, words, batch_size=BATCH_SIZE)

    # 4) JSON 저장
    save_embedding_dict(OUTPUT_FILE, embedding_dict)

    print("=== 모든 작업 완료 ===")


if __name__ == "__main__":
    main()
