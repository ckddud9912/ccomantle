from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sentence_transformers import SentenceTransformer

import fasttext

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 모델 로드 ---
ft = fasttext.load_model("cc.ko.300.bin")      # FastText 한국어 사전
sbert = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")  # SBERT 백업 모델

ANSWER = None
ANSWER_VEC = None
submissions = []

def encode_word(word: str) -> np.ndarray:
    """
    FastText 사전에 있으면 FastText vector
    없으면 SBERT vector 로 대체
    """
    try:
        vec = ft.get_word_vector(word)
        if vec is not None and np.linalg.norm(vec) > 0:
            return vec
    except:
        pass

    return sbert.encode([word], convert_to_numpy=True)[0]


def cosine_to_score(cosine: float) -> float:
    """꼬맨틀 스타일 점수 -20 ~ +70 근사"""
    score = (cosine + 0.2) * 70
    return round(float(max(-20, min(70, score))), 2)


@app.get("/")
def page_index():
    return FileResponse("static/index.html")


@app.get("/game")
def page_game():
    return FileResponse("static/game.html")


@app.get("/admin")
def page_admin():
    return FileResponse("static/admin.html")


@app.post("/set_answer")
def set_answer(word: str = Form(...)):
    global ANSWER, ANSWER_VEC, submissions
    word = word.strip()
    if not word:
        return JSONResponse({"ok": False, "error": "정답 단어를 입력하세요."})

    ANSWER = word
    ANSWER_VEC = encode_word(word)
    submissions = []
    return {"ok": True, "answer": word}


@app.get("/guess")
def guess(word: str, team: str):
    global ANSWER, ANSWER_VEC, submissions
    if ANSWER is None:
        return {"ok": False, "error": "아직 정답이 설정되지 않았습니다."}

    word = word.strip()
    team = team.strip() or "이름없는 팀"
    vec = encode_word(word)

    cos = float(np.dot(vec, ANSWER_VEC) / (np.linalg.norm(vec) * np.linalg.norm(ANSWER_VEC)))
    similarity = cosine_to_score(cos)

    submissions.append({
        "team": team,
        "word": word,
        "similarity": similarity
    })
    submissions.sort(key=lambda x: x["similarity"], reverse=True)

    return {"ok": True, "similarity": similarity}


@app.get("/leaderboard")
def leaderboard():
    return {"ok": True, "leaderboard": submissions}
