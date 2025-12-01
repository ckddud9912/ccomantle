from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return _model


def get_vector(word: str):
    if not word:
        return None
    try:
        m = load_model()
        v = m.encode([word], convert_to_numpy=True)[0]
        return v.astype(np.float32)
    except:
        return None
