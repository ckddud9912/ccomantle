import json
import os
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ===========================================
# 경로 설정
# ===========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src 기준
DATA_PATH = os.path.join(BASE_DIR, "..", "data")

WORDS_FILE = os.path.join(DATA_PATH, "words_50000.json")
OUTPUT_FILE = os.path.join(DATA_PATH, "embedding_dictionary_e5_scaled.json")

# 원본 저장 파일 (중간 저장용)
RAW_OUTPUT_FILE = os.path.join(DATA_PATH, "embedding_dictionary_e5_raw.json")

# ===========================================
# 모델 설정
# ===========================================
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_TOP1000 = 0.63               # 목표 유사도
SCALE_RANGE = np.linspace(0.5, 3.0, 40)   # 탐색 범위


# ===========================================
# 유틸 함수
# ===========================================
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
        cls_emb = outputs.last_hidden_state[:, 0, :]

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


# ===========================================
#  스케일링 단계 (TOP1000 ≈ 0.63 자동 조정)
# ===========================================
def cosine_sim(a, b):
    return float(np.dot(a, b))


def scale_embeddings(emb_dict: Dict[str, List[float]]):
    print("[정보] 스케일링 시작 (TOP1000 ≈ 0.63 목표)")

    words = list(emb_dict.keys())
    X = np.array([emb_dict[w] for w in words], dtype=np.float32)

    # ---- Mean Centering ----
    centroid = X.mean(axis=0)
    X = X - centroid

    # ---- Normalize ----
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # 기준 단어 10개 랜덤 선택 (분포 안정화)
    np.random.seed(42)
    ref_indices = np.random.choice(len(X), 10, replace=False)

    def get_top1000_sim(scale):
        Xs = X * scale
        Xs = Xs / np.linalg.norm(Xs, axis=1, keepdims=True)

        vals = []
        for ref in ref_indices:
            sims = Xs @ Xs[ref]
            sims.sort()
            vals.append(sims[-1000])
        return float(np.mean(vals))

    # ---- scale 탐색 ----
    best_scale = 1.0
    best_diff = 999

    print("[정보] 스케일 탐색 중...")
    for s in SCALE_RANGE:
        sim1000 = get_top1000_sim(s)
        diff = abs(sim1000 - TARGET_TOP1000)
        if diff < best_diff:
            best_diff = diff
            best_scale = s

    print(f"[INFO] 선택된 스케일: {best_scale:.4f} (diff={best_diff:.4f})")

    # ---- 최종 스케일 적용 ----
    final = X * best_scale
    final = final / np.linalg.norm(final, axis=1, keepdims=True)

    # dict로 다시 변환
    scaled_dict = {w: vec.tolist() for w, vec in zip(words, final)}
    return scaled_dict


# ===========================================
# 저장
# ===========================================
def save_embedding_dict(path, data: Dict[str, List[float]]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"[완료] 저장 완료: {path} ({len(data)}개 단어)")


# ===========================================
# 메인
# ===========================================
def main():
    print("=== E5/KoE5 임베딩 생성 시작 ===")

    # 1) 단어 로드
    words = load_words(WORDS_FILE)

    # 2) 모델 로드
    tokenizer, model = load_e5_model(MODEL_NAME)

    # 3) 임베딩 생성
    emb_dict_raw = build_embedding_dict(words, tokenizer, model)

    # 4) 중간 원본 저장 (선택)
    save_embedding_dict(RAW_OUTPUT_FILE, emb_dict_raw)

    # 5) 스케일링 적용 (TOP1000 ≈ 0.63)
    emb_dict_scaled = scale_embeddings(emb_dict_raw)

    # 6) 최종 저장
    save_embedding_dict(OUTPUT_FILE, emb_dict_scaled)

    print("=== 모든 작업 완료 ===")


if __name__ == "__main__":
    main()
