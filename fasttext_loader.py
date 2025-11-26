from sentence_transformers import SentenceTransformer
import numpy as np

_model = None


def load_model():
    """
    한국어용 Sentence-BERT 모델 로드.
    Hugging Face Hub에서 자동으로 받아온다.
    """
    global _model
    if _model is None:
        # 한국어 문장/단어 임베딩 모델
        _model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return _model


def get_vector(word: str):
    """
    주어진 단어(또는 짧은 표현)를 임베딩 벡터로 변환.
    문제가 생기면 None 반환해서 OOV 처리와 동일하게 취급.
    """
    if not word:
        return None

    model = load_model()
    try:
        emb = model.encode([word], convert_to_numpy=True, normalize_embeddings=False)
        # emb.shape == (1, hidden_dim)
        return np.array(emb[0], dtype=np.float32)
    except Exception:
        return None
