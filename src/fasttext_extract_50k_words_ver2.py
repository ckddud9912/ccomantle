import argparse
import json
import os
import re
import sys

import numpy as np
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")

TOP_K = 50000
MIN_LEN = 2
MAX_LEN = 5


def is_korean(word):
    return re.search(r"[가-힣]", word) is not None


def _resolve(cli_val, env_key, label):
    path = cli_val or os.getenv(env_key, "")
    if not path:
        sys.exit(f"[ERROR] {label} 경로 필요: --model 또는 환경변수 {env_key}")
    return path


def parse_args():
    p = argparse.ArgumentParser(description="FastText .bin → embedding_dictionary.json (PCA 128)")
    p.add_argument("--model", default=None, help="FastText .bin 경로 (또는 FASTTEXT_MODEL_PATH)")
    p.add_argument("--output", default=os.path.join(DATA_PATH, "embedding_dictionary.json"))
    return p.parse_args()


def main():
    args = parse_args()
    model_path = _resolve(args.model, "FASTTEXT_MODEL_PATH", "FastText .bin 모델")

    import fasttext
    print("FastText 모델 로드 중...")
    model = fasttext.load_model(model_path)
    print("모델 로드 완료!")

    words = model.get_words()
    valid = [w for w in words if MIN_LEN <= len(w) <= MAX_LEN and is_korean(w)]
    print(f"한국어 후보 단어: {len(valid)}개")

    final_words = valid[:TOP_K]
    print(f"최종 {TOP_K}개 단어 선택 완료")

    print("FastText 벡터 생성 중...")
    vectors = np.array([model.get_word_vector(w) for w in final_words], dtype=np.float32)
    print(f"벡터 생성 완료: shape={vectors.shape}")

    print("PCA 차원 축소 (128)")
    pca = PCA(n_components=128, random_state=42)
    vectors_reduced = pca.fit_transform(vectors)
    print(f"PCA 완료: shape={vectors_reduced.shape}")

    norms = np.linalg.norm(vectors_reduced, axis=1, keepdims=True)
    vectors_normalized = vectors_reduced / norms
    print("L2 정규화 완료")

    emb_dict = {w: vec.tolist() for w, vec in zip(final_words, vectors_normalized)}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(emb_dict, f, ensure_ascii=False, indent=2)
    print(f"저장 완료 → {args.output}")


if __name__ == "__main__":
    main()
