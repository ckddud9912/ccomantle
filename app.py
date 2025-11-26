from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SBERT 한국어 모델
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

ANSWER = None
ANSWER_VEC = None
submissions = []


def encode_vector(word: str) -> np.ndarray:
    return model.encode([word], convert_to_numpy=True)[0]


def cosine_to_ccomantle_score(cosine: float) -> float:
    """
    cosine(-1~1) → 꼬맨틀 점수 (-20 ~ +70)
    """
    score = (cosine * 45) + 25
    return round(max(-20.0, min(70.0, score)), 2)


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/game")
def game():
    return FileResponse("static/game.html")


@app.get("/admin")
def admin():
    return FileResponse("static/admin.html")


@app.post("/set_answer")
def set_answer(word: str = Form(...)):
    global ANSWER, ANSWER_VEC, submissions

    word = word.strip()
    if not word:
        return JSONResponse({"ok": False, "error": "정답을 입력하세요."})

    ANSWER = word
    ANSWER_VEC = encode_vector(word)
    submissions = []  # 새 라운드

    return JSONResponse({"ok": True, "answer": ANSWER})


@app.get("/guess")
def guess(word: str, team: str):
    global ANSWER, ANSWER_VEC, submissions

    if ANSWER is None:
        return JSONResponse({"ok": False, "error": "아직 정답이 설정되지 않았습니다."})

    word = word.strip()
    team = (team or "").strip() or "이름없는 팀"

    if not word:
        return JSONResponse({"ok": False, "error": "단어를 입력하세요."})

    if word == ANSWER:
        similarity = 70.0
    else:
        vec = encode_vector(word)
        cosine = float(np.dot(vec, ANSWER_VEC) /
                       (np.linalg.norm(vec) * np.linalg.norm(ANSWER_VEC)))
        similarity = cosine_to_ccomantle_score(cosine)

    submissions.append({"team": team, "word": word, "similarity": similarity})
    submissions.sort(key=lambda x: x["similarity"], reverse=True)

    return JSONResponse({"ok": True, "similarity": similarity})


@app.get("/leaderboard")
def leaderboard():
    return JSONResponse({"ok": True, "leaderboard": submissions})
