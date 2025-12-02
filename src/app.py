import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Optional
import orjson  # 빠른 JSON 로딩

# ==================================================
# DIAGNOSTIC
# ==================================================
print("[DIAG 1] Python interpreter started and os module is working.")

# ==================================================
# 전역 변수 선언 (로드 전 None)
# ==================================================
embedding_dict: Optional[Dict[str, List[float]]] = None
words_list: Optional[List[str]] = None
word_to_rank: Optional[Dict[str, int]] = None

answer_word: Optional[str] = None
answer_vector: Optional[np.ndarray] = None
submissions: List[Dict] = []

sim_top1 = None
sim_top20 = None
sim_top1000 = None

# ==================================================
# 디렉토리 정의
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

print(f"[DIAG 2] BASE_DIR: {BASE_DIR}")
print(f"[DIAG 2] DATA_DIR: {DATA_DIR}")
print(f"[DIAG 2] STATIC_DIR: {STATIC_DIR}")

# ==================================================
# FastAPI 앱 생성
# ==================================================
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==================================================
# STARTUP EVENT: 임베딩 및 단어 리스트 로드
# ==================================================
@app.on_event("startup")
async def load_embeddings():
    print("[STARTUP] Begin loading embedding dictionary...")
    global embedding_dict, words_list, word_to_rank
    global sim_top1, sim_top20, sim_top1000

    embedding_path = os.path.join(DATA_DIR, "embedding_dictionary.json")
    words_path = os.path.join(DATA_DIR, "words_50000.json")

    print("[DIAG 3] Loading embedding dictionary...")
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"{embedding_path} not found!")

    if not os.path.exists(words_path):
        raise FileNotFoundError(f"{words_path} not found!")

    with open(embedding_path, "rb") as f:
        embedding_dict = orjson.loads(f.read())

    with open(words_path, "rb") as f:
        words_list = orjson.loads(f.read())

    word_to_rank = {w: i for i, w in enumerate(words_list)}

    print(f"[INFO] Loaded {len(words_list)} words and embeddings.")


# ==================================================
# COSINE SIMILARITY (정규화된 버전)
# ==================================================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # cosine_sim() 함수는 현재 단순 dot product만 하고 있음. 일반적인 cosine similarity라면 벡터 정규화 필요:
    # 이걸 적용하면 similarity 계산이 더 일반적인 의미의 cosine similarity가 됨. (현재는 단순 내적 값)

    if a is None or b is None:
        return -1.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


"""
1. np.linalg.norm(a) → 벡터 a의 길이(유클리디안 노름).
2. 내적(np.dot(a, b))을 벡터 길이의 곱으로 나누면 정규화된 cosine similarity.
3. 벡터 길이가 0이면 유사도 계산 불가이므로 -1.0 반환.
"""


# ==================================================
# ANSWER RANKING 계산
# ==================================================


def compute_answer_ranking():
    global answer_vector, word_to_rank, sim_top1, sim_top20, sim_top1000

    if embedding_dict is None or words_list is None:
        return

    sims = []
    for w in words_list:
        vec = np.array(embedding_dict[w])
        sims.append((w, cosine_sim(answer_vector, vec)))

    sims.sort(key=lambda x: x[1], reverse=True)

    word_to_rank.clear()
    for idx, (w, _) in enumerate(sims):
        word_to_rank[w] = idx + 1

    sim_top1 = sims[0][1]
    sim_top20 = sims[19][1] if len(sims) >= 20 else sims[-1][1]
    sim_top1000 = sims[999][1] if len(sims) >= 1000 else sims[-1][1]

    print("[INFO] 정답 기준 랭킹 테이블 생성 완료")


# ==================================================
# REQUEST MODELS
# ==================================================
class SetAnswerRequest(BaseModel):
    answer: str


class GuessRequest(BaseModel):
    team: str
    word: str


# ==================================================
# ENDPOINTS
# ==================================================
@app.post("/set_answer")
def set_answer(req: SetAnswerRequest):
    global answer_word, answer_vector, submissions, words_list

    if embedding_dict is None:
        return JSONResponse({"error": "Embedding dictionary not loaded yet"})

    if req.answer not in embedding_dict:
        return JSONResponse(
            {"error": f"정답이 사전 단어에 없습니다. 단어 수: {len(embedding_dict)}"}
        )

    answer_word = req.answer
    answer_vector = np.array(embedding_dict[answer_word])
    submissions = []

    # 전역 words_list를 embedding_dict 기준으로 필터링
    words_list = [w for w in words_list if w in embedding_dict]

    compute_answer_ranking()
    return {"status": "ok", "answer": answer_word}


@app.post("/guess")
def guess(req: GuessRequest):
    global submissions

    if answer_word is None:
        return JSONResponse({"error": "정답이 아직 설정되지 않았습니다."})

    team = req.team.strip()
    word = req.word.strip()

    if len(word) == 0:
        return JSONResponse({"error": "단어가 비어 있습니다."})

    # 중복 제출 방지
    for s in submissions:
        if s["team"] == team and s["word"] == word:
            return {"result": "duplicate"}

    if word == answer_word:
        entry = {
            "team": team,
            "word": word,
            "is_answer": True,
            "rank": 1,
            "similarity": 1.0,
        }
        submissions.append(entry)
        return {"result": "correct", "entry": entry}

    # 벡터 유사도 계산
    if embedding_dict is not None and word in embedding_dict:
        vec = np.array(embedding_dict[word])
        similarity = round(float(cosine_sim(answer_vector, vec)), 3)
        rank = word_to_rank.get(word, None)
    else:
        similarity = -1.0
        rank = None

    entry = {
        "team": team,
        "word": word,
        "is_answer": False,
        "rank": rank if rank is not None else "???",
        "similarity": similarity,
    }

    submissions.append(entry)
    return {"result": "ok", "entry": entry}


@app.get("/leaderboard")
def leaderboard():
    sorted_submissions = sorted(submissions, key=lambda x: (-x["similarity"]))
    return {
        "submissions": sorted_submissions,
        "answer": answer_word,
        "sim_top1": sim_top1,
        "sim_top20": sim_top20,
        "sim_top1000": sim_top1000,
    }


# ==================================================
# STATIC PAGES
# ==================================================
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/game", response_class=HTMLResponse)
def game_page():
    with open(os.path.join(STATIC_DIR, "game.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    with open(os.path.join(STATIC_DIR, "admin.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
