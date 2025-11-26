import fasttext
import numpy as np

model = None

def load_model():
    global model
    if model is None:
        model = fasttext.load_model("cc.ko.300.bin")
    return model

def get_vector(word: str):
    m = load_model()
    try:
        v = m.get_word_vector(word)
        return np.array(v)
    except:
        return None
