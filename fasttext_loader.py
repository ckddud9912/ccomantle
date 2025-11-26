import fasttext
import numpy as np
from typing import Optional

# 전역 상태
_model = None
_vocab = None   # FastText 내부 단어 목록 set (OOV 판별용)


def _load_model():
    """
    FastText 모델을 지연 로딩(lazy loading) 방식으로 초기화.
    - HuggingFace Spaces health check 단계에서는 호출되지 않도록
      import 시점에는 로딩하지 않음.
    """
    global _model
    if _model is None:
        # 모델 파일 이름은 HF Spaces에 업로드된 파일명과 동일해야 함
        _model = fasttext.load_model("cc.ko.300.bin")
    return _model


def _ensure_vocab():
    """
    FastText 모델의 vocabulary 를 set 으로 만들어서
    OOV(없는 단어) 판별에 사용.
    """
    global _vocab
    if _vocab is None:
        m = _load_model()
        # get_words() 는 모델의 단어 목록을 반환
        words = m.get_words()
        _vocab = set(words)
    return _vocab


def get_vector(word: str) -> Optional[np.ndarray]:
    """
    주어진 단어의 벡터를 반환.
    - 모델 vocabulary 에 단어가 없으면(None) → OOV 처리
    - 있으면 numpy 배열로 반환
    """
    if not word:
        return None

    m = _load_model()
    vocab = _ensure_vocab()

    if word not in vocab:
        return None

    try:
        vec = m.get_word_vector(word)
        return np.array(vec, dtype="float32")
    except Exception:
        # 예외 발생 시에도 OOV처럼 처리
        return None
