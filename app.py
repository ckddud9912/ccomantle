from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

ANSWER = None
ANSWER_VEC = None
submissions = []
SIM_STATS = {}     # ← Top1, Top10, Top1000 유사도 저장


def cosine_sim(v1, v2):
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return 0.0 if norm == 0 else dot / norm


def calculate_similarity_stats(answer_word):
    """정답과 전체 단어 중 임의의 샘플로 유사도 통계 생성"""
    sample_words = ["사과", "자동차", "고양이", "행복", "바다", "학교", "음악", "사랑", "여행",
                    "책상", "노트북", "커피", "강아지", "의자", "하늘", "달", "겨울", "여름"]

    vecs = model.encode(sample_words, convert_to_numpy=True)

    sims = []
    for word, vec in zip(sample_words, vecs):
        sims.append((word, cosine_sim(vec, ANSWER_VEC)))

    sims.sort(key=lambda x: x[1], reverse=True)

    top1 = round(sims[0][1] * 100, 2)
    top10 = round(sims[min(9, len(sims)-1)][1] * 100, 2)
    top1000 = round(sims[min(999, len(sims)-1)][1] * 100, 2)

    return {"top1": top1, "top10": top10, "top1000": top1000}


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
    global ANSWER, ANSWER_VEC, submissions, SIM_STATS

    ANSWER = word.strip()
    ANSWER_VEC = model.encode([ANSWER], convert_to_numpy=True)[0]
    submissions = []  # reset scoreboard

    SIM_STATS = calculate_similarity_stats(ANSWER)

    return {
        "ok": True,
        "answer": ANSWER,
        "stats": SIM_STATS
    }


@app.get("/guess")
def guess(word: str, team: str):
    global ANSWER, ANSWER_VEC

    if ANSWER is None:
        return {"ok": False, "error": "아직 진행자가 정답을 설정하지 않았습니다."}

    vec = model.encode([word], convert_to_numpy=True)[0]
    sim_raw = cosine_sim(vec, ANSWER_VEC)
    sim_scaled = max(0, sim_raw * 100)
    sim_scaled = round(sim_scaled, 2)

    submissions.append({
        "team": team,
        "word": word,
        "similarity": sim_scaled,
    })

    return {"ok": True, "similarity": sim_scaled}


@app.get("/leaderboard")
def leaderboard():

    sorted_list = sorted(submissions, key=lambda x: x["similarity"], reverse=True)

    for idx, row in enumerate(sorted_list):
        rank = idx + 1
        row["rank"] = rank if rank <= 999 else "1000위 이상"

    return {"ok": True, "leaderboard": sorted_list, "stats": SIM_STATS}
