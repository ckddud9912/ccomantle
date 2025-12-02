import json
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src 폴더 기준
DATA_PATH = os.path.join(BASE_DIR, "..", "data")


WORDS_FILE = os.path.join(DATA_PATH, "words_50000.json")  # ← 5만 단어 리스트 사용
OUTPUT_FILE = os.path.join(DATA_PATH, "embedding_dictionary.json")
MODEL_NAME = "jhgan/ko-sroberta-multitask"
BATCH_SIZE = 64


def load_words(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 파일 없음")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} 내용이 리스트가 아님")

    # 중복 제거
    seen = set()
    unique_words = []
    for w in data:
        if isinstance(w, str) and w not in seen:
            seen.add(w)
            unique_words.append(w)

    print(f"[정보] 총 단어 {len(data)}개 → 중복 제거 후 {len(unique_words)}개")
    return unique_words


def load_model(model_name: str) -> SentenceTransformer:
    print(f"[정보] 모델 로드 중: {model_name}")
    model = SentenceTransformer(model_name)
    print("[정보] 모델 로드 완료")
    return model


def compute_embeddings(model, words, batch_size=64):
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

    print(f"[정보] 임베딩 완료: {len(words)}개, 차원 {embeddings.shape[1]}")

    emb_dict = {}
    for w, vec in zip(words, embeddings):
        emb_dict[w] = vec.tolist()

    return emb_dict


def save_embedding_dict(path, data):
    if os.path.exists(path):
        print(f"[경고] 기존 {path} 덮어쓰기")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[완료] 저장 완료: {path} ({len(data)}개 단어)")


def main():
    print("=== 50,000 단어 임베딩 생성 시작 ===")

    words = load_words(WORDS_FILE)
    model = load_model(MODEL_NAME)

    emb_dict = compute_embeddings(model, words, batch_size=BATCH_SIZE)
    save_embedding_dict(OUTPUT_FILE, emb_dict)

    print("=== 모든 작업 완료 ===")


if __name__ == "__main__":
    main()
