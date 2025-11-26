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

# ---- SBERT 한국어 모델 (허깅페이스에서 자동 다운로드) ----
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

ANSWER = None          # 정답 단어 (문자열)
ANSWER_VEC = None      # 정답 임베딩 벡터
submissions = []       # {team, word, similarity} 리스트


def encode_word(word: str) -> np.ndarray:
    """SBERT 임베딩 생성"""
    return model.encode([word], convert_to_numpy=True)[0]


def cosine_to_score(cosine: float) -> float:
    """
    cosine(-1~1) → 점수(-20 ~ 70)로 변환
    원조 꼬맨틀 느낌 나도록 대략적인 스케일 조정
    """
    score = (cosine + 0.2) * 70
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


# ---------- 진행자: 정답 설정 ----------
@app.post("/set_answer")
def set_answer(word: str = Form(...)):
    global ANSWER, ANSWER_VEC, submissions

    word = word.strip()
    if not word:
        return JSONResponse({"ok": False, "error": "정답 단어를 입력하세요."})

    ANSWER = word
    ANSWER_VEC = encode_word(word)

    # 새 라운드 시작 → 기존 리더보드 초기화
    submissions = []

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

    # 정답인 경우
    if word == ANSWER:
        similarity = 70.0  # 스케일 상 최댓값으로 처리
    else:
        vec = encode_word(word)
        cosine = float(np.dot(vec, ANSWER_VEC) /
                       (np.linalg.norm(vec) * np.linalg.norm(ANSWER_VEC)))
        similarity = cosine_to_score(cosine)

    submissions.append({
        "team": team,
        "word": word,
        "similarity": similarity,
    })

    # 유사도 높은 순으로 정렬
    submissions.sort(key=lambda x: x["similarity"], reverse=True)

    return JSONResponse({"ok": True, "similarity": similarity})


# ---------- 리더보드 ----------
@app.get("/leaderboard")
def leaderboard():
    return JSONResponse({"ok": True, "leaderboard": submissions})
