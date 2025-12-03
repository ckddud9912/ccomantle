import json
import os
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src 기준
DATA_PATH = os.path.join(BASE_DIR, "..", "data")

WORDS_FILE = os.path.join(DATA_PATH, "words_50000.json")
OUTPUT_FILE = os.path.join(DATA_PATH, "embedding_dictionary_e5.json")

###############################################
# ★ 원하는 모델 선택 (E5/KoE5 지원)
###############################################
MODEL_NAME = "intfloat/multilingual-e5-large"
# MODEL_NAME = "BM-K/KoE5"   # 한국어 최적
# MODEL_NAME = "intfloat/e5-large-v2"
###############################################

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_words(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 파일 없음")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 리스트 아님")

    seen = set()
    unique_words = []
    for w in data:
        if isinstance(w, str) and w not in seen:
            seen.add(w)
            unique_words.append(w)

    print(f"[정보] 단어 {len(data)}개 → 중복 제거 후 {len(unique_words)}개")
    return unique_words


def load_e5_model(model_name: str):
    print(f"[정보] 모델 로딩: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    print("[정보] 모델 로딩 완료")
    return tok, model


def encode_e5_batch(tokenizer, model, texts: List[str]) -> np.ndarray:
    """
    E5 / KoE5 전용 batch encode
    - "query: " prefix 자동 추가
    - CLS 벡터 추출
    - L2 normalize
    """
    # E5는 반드시 prefix를 붙여야 함
    e5_inputs = [f"query: {t}" for t in texts]

    inputs = tokenizer(
        e5_inputs,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS 토큰

    # L2 정규화
    cls_emb = cls_emb / cls_emb.norm(dim=1, keepdim=True)

    return cls_emb.cpu().numpy()


def build_embedding_dict(words: List[str], tokenizer, model):
    print("[정보] 임베딩 계산 시작")
    emb_dict: Dict[str, List[float]] = {}

    for i in tqdm(range(0, len(words), BATCH_SIZE)):
        batch_words = words[i:i + BATCH_SIZE]
        vectors = encode_e5_batch(tokenizer, model, batch_words)

        for w, vec in zip(batch_words, vectors):
            emb_dict[w] = vec.tolist()

    print(f"[정보] 총 {len(emb_dict)}개 단어 임베딩 완료")
    return emb_dict


def save_embedding_dict(path: str, data: Dict[str, List[float]]):
    if os.path.exists(path):
        print(f"[경고] 기존 파일 덮어쓰기: {path}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"[완료] 저장 완료: {path}")


def main():
    print("=== E5/KoE5 임베딩 생성 시작 ===")

    words = load_words(WORDS_FILE)
    tokenizer, model = load_e5_model(MODEL_NAME)

    emb_dict = build_embedding_dict(words, tokenizer, model)
    save_embedding_dict(OUTPUT_FILE, emb_dict)

    print("=== 모든 작업 완료 ===")


if __name__ == "__main__":
    main()
