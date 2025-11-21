from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# CORS (혹시 모를 문제 방지용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- SBERT 모델 로드 ----
# 한국어 SBERT 모델 하나 선택 (무료)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

ANSWER = None          # 정답 단어 (문자열)
ANSWER_VEC = None      # 정답 임베딩 (벡터)
submissions = []       # 모든 팀의 제출 기록


def cosine_sim(v1, v2):
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm == 0:
        return 0.0
    return dot / norm


@app.get("/")
def page_index():
    return FileResponse("static/index.html")


@app.get("/game")
def page_game():
    return FileResponse("static/game.html")


@app.get("/admin")
def page_admin():
    return FileResponse("static/admin.html")


# 진행자: 정답 설정
@app.post("/set_answer")
def set_answer(word: str = Form(...)):
    global ANSWER, ANSWER_VEC, submissions

    word = word.strip()
    if not word:
        return JSONResponse({"ok": False, "error": "정답 단어를 입력하세요."})

    # 정답 임베딩 만들기
    ANSWER = word
    ANSWER_VEC = model.encode([ANSWER], convert_to_numpy=True)[0]

    # 새 라운드 시작 → 기존 제출 기록 초기화
    submissions = []

    return JSONResponse({"ok": True, "answer": ANSWER})


# 팀: 단어 제출
@app.get("/guess")
def guess(word: str, team: str):
    global ANSWER, ANSWER_VEC, submissions

    if ANSWER is None or ANSWER_VEC is None:
        return {"ok": False, "error": "아직 진행자가 정답을 설정하지 않았습니다."}

    word = word.strip()
    team = team.strip() or "이름없는 팀"

    if not word:
        return {"ok": False, "error": "단어를 입력하세요."}

    # 정답인 경우
    if word == ANSWER:
        similarity = 100.0
        correct = True
    else:
        # SBERT 유사도 계산 (보통 0~1 사이 → 0~100 점수로 스케일)
        vec = model.encode([word], convert_to_numpy=True)[0]
        sim_raw = cosine_sim(vec, ANSWER_VEC)  # 대략 0~1
        similarity = max(0.0, float(sim_raw) * 100.0)
        similarity = round(similarity, 2)
        correct = False

    submissions.append({
        "team": team,
        "word": word,
        "similarity": similarity,
    })

    return {
        "ok": True,
        "correct": correct,
        "similarity": similarity,
    }


# 리더보드: 유사도 높은 순
@app.get("/leaderboard")
def leaderboard():
    # similarity 높은 순으로 정렬
    sorted_list = sorted(
        submissions,
        key=lambda x: x["similarity"],
        reverse=True
    )
    return {"ok": True, "leaderboard": sorted_list}
