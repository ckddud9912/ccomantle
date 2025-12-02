import fasttext
import json
import re
import numpy as np
from sklearn.decomposition import PCA
import os

# 🔥 FastText 모델 경로
MODEL_PATH = r"C:\Users\창영\Desktop\cc.ko.300.bin"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src 폴더 기준
DATA_PATH = os.path.join(BASE_DIR, "..", "data")

# 🔹 출력 파일
OUTPUT_FILE = DATA_PATH + "/embedding_dictionary.json"

# 🔹 top-k 필터링 (예: 50,000개)
TOP_K = 50000

# 🔹 단어 필터 조건
MIN_LEN = 2
MAX_LEN = 5

def is_korean(word):
    return re.search(r"[가-힣]", word) is not None

def main():
    print("🔵 FastText 모델 로드 중...")
    model = fasttext.load_model(MODEL_PATH)
    print("✔ 모델 로드 완료!")

    print("🔵 단어 후보 추출 중...")
    words = model.get_words()
    valid = [w for w in words if MIN_LEN <= len(w) <= MAX_LEN and is_korean(w)]
    print(f"한국어 후보 단어: {len(valid)}개")

    # top-k 단어 선택
    final_words = valid[:TOP_K]
    print(f"✔ 최종 {TOP_K}개 단어 선택 완료")

    print("🔵 FastText 벡터 생성 중...")
    vectors = np.array([model.get_word_vector(w) for w in final_words], dtype=np.float32)
    print(f"벡터 생성 완료: shape={vectors.shape}")

    print("🔵 PCA 차원 축소 (원하면 차원 변경 가능)")
    pca = PCA(n_components=128, random_state=42)
    vectors_reduced = pca.fit_transform(vectors)
    print(f"PCA 완료: shape={vectors_reduced.shape}")

    print("🔵 L2 정규화")
    norms = np.linalg.norm(vectors_reduced, axis=1, keepdims=True)
    vectors_normalized = vectors_reduced / norms
    print("✔ 정규화 완료")

    # 🔹 embedding_dictionary.json 생성
    emb_dict = {w: vec.tolist() for w, vec in zip(final_words, vectors_normalized)}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(emb_dict, f, ensure_ascii=False, indent=2)
    print(f"🎉 {OUTPUT_FILE} 생성 완료!")

if __name__ == "__main__":
    main()
