from gensim.models.fasttext import FastTextKeyedVectors
import numpy as np

_model = None
_vocab = None


def load_model():
    global _model
    if _model is None:
        # gensim FastTextKeyedVectors 는 FastText 바이너리 모델을 그대로 로드 가능
        _model = FastTextKeyedVectors.load("cc.ko.300.bin")
    return _model


def ensure_vocab():
    global _vocab
    if _vocab is None:
        model = load_model()
        _vocab = set(model.key_to_index.keys())
    return _vocab


def get_vector(word: str):
    if not word:
        return None

    model = load_model()
    vocab = ensure_vocab()

    if word not in vocab:
        return None

    try:
        vec = model.get_vector(word)
        return np.array(vec, dtype=np.float32)
    except:
        return None
