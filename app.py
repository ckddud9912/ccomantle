from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sentence_transformers import SentenceTransformer

from fasttext_loader import get_vector   # ← 맨 위에 추가

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- SBERT 한국어 모델 (사용은 최소화, fallback 용) ----
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

ANSWER = None
ANSWER_VEC = None
submissions = []


@app.get("/")
def page_index():
    return FileResponse("static/index.html")


@app.get("/game")
def page_game():
    return FileResponse("static/game.html")


@app.get("/admin")
def page_admin():
    return FileResponse("static/admin.html")


# ---------- 진행자: 정답 설정 ----------
@app.post("/set_answer")
def set_answer(word: str = Form(...)):
    global ANSWER, ANSWER_VEC, submissions

    word = word.strip()
    if not word:
        return JSONResponse({"ok": False, "error": "정답 단어를 입력하세요."})

    ANSWER = word
    ANSWER_VEC = get_vector(ANSWER)   # FastText 기반 벡터 생성

    submissions = []  # 라운드 리셋

    return JSONResponse({"ok": True, "answer": ANSWER})


# ---------- 참가자: 단어 제출 ----------
@app.get("/guess")
def guess(word: str, team: str):
    global ANSWER, ANSWER_VEC, submissions

    if ANSWER is None or ANSWER_VEC is None:
        return JSONResponse({"ok": False, "error": "아직 진행자가 정답을 설정하지 않았습니다."})

    word = word.strip()
    team = (team or "").strip() or "이름없는 팀"

    if not word:
        return JSONResponse({"ok": False, "error": "단어를 입력하세요."})

    # 정답
    if word == ANSWER:
        similarity = 70.0
    else:
        vec = get_vector(word)
        if vec is None:
            similarity = -20.0
        else:
            cosine = float(np.dot(vec, ANSWER_VEC) /
                           (np.linalg.norm(vec) * np.linalg.norm(ANSWER_VEC)))
            similarity = round((cosine * 45) + 25, 2)  # (-20 ~ 70 스케일)

    submissions.append({
        "team": team,
        "word": word,
        "similarity": similarity,
    })

    submissions.sort(key=lambda x: x["similarity"], reverse=True)

    return JSONResponse({"ok": True, "similarity": similarity})


# ---------- 리더보드 ----------
@app.get("/leaderboard")
def leaderboard():
    return JSONResponse({"ok": True, "leaderboard": submissions})
