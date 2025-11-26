from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 한국어 SBERT 임베딩 모델
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

ANSWER = None                # 정답 단어
ANSWER_VEC = None            # 정답 벡터
submissions = []             # 제출 기록

# 유사도 안내 문구용 기준값
BEST_SCORE = None            # 가장 유사한 단어 점수(1위)
TOP10_SCORE = None           # 10위
TOP1000_SCORE = None         # 1000위


def encode(word: str):
    """SBERT 임베딩"""
    return model.encode([word], convert_to_numpy=True)[0]


def cosine_to_score(cosine: float):
    """
    SBERT cosine (-1 ~ 1) → 꼬맨틀 감성 점수(-20 ~ 70)
    """
    score = (cosine * 45) + 25
    score = max(-20.0, min(70.0, score))
    return round(float(score), 2)


@app.get("/")
def page_index():
    return FileResponse("static/index.html")


@app.get("/game")
def page_game():
    return FileResponse("static/game.html")


@app.get("/admin")
def page_admin():
    return FileResponse("static/admin.html")


# ---------------- 정답 설정 ----------------
@app.post("/set_answer")
def set_answer(word: str = Form(...)):
    global ANSWER, ANSWER_VEC, submissions
    global BEST_SCORE, TOP10_SCORE, TOP1000_SCORE

    word = word.strip()
    if not word:
        return {"ok": False, "error": "정답 단어를 입력하세요."}

    try:
        vec = encode(word)
    except Exception:
        return {"ok": False, "error": "임베딩 생성 실패 — 너무 특수한 단어일 수 있습니다."}

    ANSWER = word
    ANSWER_VEC = vec
    submissions = []

    # 원조 꼬맨틀 느낌으로 고정 안내값 제공
    BEST_SCORE = 70.0
    TOP10_SCORE = 45.0
    TOP1000_SCORE = 15.0

    return {"ok": True, "answer": ANSWER}


# ---------------- 참가자 제출 ----------------
@app.get("/guess")
def guess(word: str, team: str):
    global ANSWER, ANSWER_VEC, submissions

    if ANSWER is None or ANSWER_VEC is None:
        return {"ok": False, "error": "아직 정답이 설정되지 않았습니다."}

    word = word.strip()
    team = (team or "").strip() or "이름없는 팀"

    if not word:
        return {"ok": False, "error": "단어를 입력하세요."}

    if word == ANSWER:
        similarity = 70.0
    else:
        try:
            vec = encode(word)
        except Exception:
            similarity = -20.0
        else:
            cosine = float(np.dot(vec, ANSWER_VEC) /
                           (np.linalg.norm(vec) * np.linalg.norm(ANSWER_VEC)))
            similarity = cosine_to_score(cosine)

    submissions.append({"team": team, "word": word, "similarity": similarity})
    submissions.sort(key=lambda x: x["similarity"], reverse=True)

    return {"ok": True, "similarity": similarity}


# ---------------- 유사도 안내 수치 ----------------
@app.get("/scoreinfo")
def scoreinfo():
    return {
        "best": BEST_SCORE,
        "top10": TOP10_SCORE,
        "top1000": TOP1000_SCORE
    }


# ---------------- 리더보드 ----------------
@app.get("/leaderboard")
def leaderboard():
    return {"ok": True, "leaderboard": submissions}
