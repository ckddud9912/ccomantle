import fasttext
import numpy as np
import hashlib

# 📌 FastText 모델 로드 (한국어)
MODEL_PATH = "cc.ko.300.bin"
model = fasttext.load_model(MODEL_PATH)

# 벡터 캐싱 (RAM 절약용)
_vec_cache = {}

def get_vector(word: str):
    word = word.strip()
    if word == "":
        return None

    # 캐싱되어 있으면 반환
    if word in _vec_cache:
        return _vec_cache[word]

    # FastText가 단어가 없더라도 subword 기반으로 생성해줌 → OK
    try:
        vec = model.get_word_vector(word)
    except Exception:
        # 최악의 경우 해싱 기반으로 fallback
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        vec = rng.normal(0, 1, 300).astype(np.float32)

    _vec_cache[word] = vec
    return vec
