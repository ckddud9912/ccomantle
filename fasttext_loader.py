from sentence_transformers import SentenceTransformer
import numpy as np

_model = None


def load_model():
    """
    한국어용 Sentence-BERT 모델 로드.
    Hugging Face Hub에서 'jhgan/ko-sroberta-multitask' 모델을 사용한다.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return _model


def get_vector(word: str):
    """
    단어(또는 짧은 문장)을 임베딩 벡터(np.ndarray)로 변환.
    문제가 생기면 None을 반환하여 OOV처럼 처리되도록 한다.
    """
    if not word:
        return None

    model = load_model()
    try:
        emb = model.encode([word], convert_to_numpy=True)
        # emb.shape == (1, hidden_dim)
        return np.array(emb[0], dtype=np.float32)
    except Exception:
        return None
