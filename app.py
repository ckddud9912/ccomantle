from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SBERT 모델
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

ANSWER = None
ANSWER_VEC = None
submissions = []
WORD_LIST = []
WORD_VECS = None


# -----------------------------
# 🔥 1) SBERT 단어 리스트 불러오기
# -----------------------------
def load_word_list():
    global WORD_LIST, WORD_VECS
    with open("static/ko_words.json", "r", encoding="utf-8") as f:
        WORD_LIST = json.load(f)

    WORD_VECS = model.encode(WORD_LIST, convert_to_numpy=True)


load_word_list()


# -----------------------------
# 🔥 2) 유사도 스케일 변환 (-20 ~ +70)
# -----------------------------
def convert_similarity(cos_sim: float):
    return round(((cos_sim + 1) / 2) * 90 - 20, 2)


# -----------------------------
# 🔥 3) 유사도 순위 계산 함수
# -----------------------------
def get_similarity_rank(word_vec):
    sims = np.dot(WORD_VECS, word_vec) / (
        np.linalg.norm(WORD_VECS, axis=1) * np.linalg.norm(word_vec)
    )

    sorted_idx = np.argsort(-sims)  # 높은 순
    return sorted_idx, sims


# -----------------------------
# 정답 설정
# -----------------------------
@app.post("/set_answer")
def set_answer(word: str = Form(...)):
    global ANSWER, ANSWER_VEC, submissions

    ANSWER = word.strip()
    ANSWER_VEC = model.encode([ANSWER], convert_to_numpy=True)[0]
    submissions = []

    sorted_idx, sims = get_similarity_rank(ANSWER_VEC)

    top1 = convert_similarity(sims[sorted_idx[0]])
    top10 = convert_similarity(sims[sorted_idx[min(9, len(sims)-1)]])
    top1000 = convert_similarity(sims[sorted_idx[min(999, len(sims)-1)]])

    return {
        "ok": True,
        "answer": ANSWER,
        "stats": {
            "top1": top1,
            "top10": top10,
            "top1000": top1000,
        }
    }


# -----------------------------
# 단어 제출
# -----------------------------
@app.get("/guess")
def guess(word: str, team: str):

    vec = model.encode([word], convert_to_numpy=True)[0]
    sim_raw = float(np.dot(vec, ANSWER_VEC) /
                    (np.linalg.norm(vec) * np.linalg.norm(ANSWER_VEC)))

    sim_scaled = convert_similarity(sim_raw)

    submissions.append({
        "team": team,
        "word": word,
        "similarity": sim_scaled,
        "vec": vec.tolist()
    })

    return {"ok": True, "similarity": sim_scaled}


# -----------------------------
# 리더보드
# -----------------------------
@app.get("/leaderboard")
def leaderboard():
    rows = []
    for row in submissions:
        vec = np.array(row["vec"])
        _, sims = get_similarity_rank(vec)

        rank = int(np.where(np.sort(-sims) == -(np.max(sims)))[0][0]) + 1
        rank = rank if rank <= 999 else "1000위 이상"

        rows.append({
            "team": row["team"],
            "word": row["word"],
            "similarity": row["similarity"],
            "rank": rank
        })

    sorted_rows = sorted(rows, key=lambda x: x["similarity"], reverse=True)
    return {"ok": True, "leaderboard": sorted_rows}
