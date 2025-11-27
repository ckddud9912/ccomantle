import run_embed
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import json
import os
from typing import List, Dict, Optional


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================
#  1) GLOBAL STATE
# ============================================================

answer_word: Optional[str] = None
answer_vector: Optional[np.ndarray] = None
submissions: List[Dict] = []

embedding_dict: Dict[str, List[float]] = {}
words_list: List[str] = []
word_to_rank: Dict[str, int] = {}

sim_top1 = None
sim_top20 = None
sim_top1000 = None


# ============================================================
#  2) LOAD EMBEDDING DICTIONARY & WORD LIST
# ============================================================

def load_embedding_dictionary():
    global embedding_dict, words_list
    if not os.path.exists("embedding_dictionary.json"):
        print("embedding_dictionary.json 파일이 없습니다.")
        return

    with open("embedding_dictionary.json", "r", encoding="utf-8") as f:
        embedding_dict = json.load(f)

    words_list = list(embedding_dict.keys())
    print(f"[INFO] embedding_dictionary.json 로드 완료 — {len(words_list)}개 단어")


load_embedding_dictionary()


# ============================================================
#  3) UTIL — COSINE 유사도
# ============================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ============================================================
#  4) UTIL — 점수 스케일링(S-curve)
# ============================================================

def score_from_cosine(cos: float) -> float:
    sim = (cos + 1) / 2
    curved = sim ** 4.2
    score = curved * 120 - 40
    return max(-100, min(100, score))


# ============================================================
#  5) MODEL — 정답 설정 시 랭킹 테이블 생성
# ============================================================

def compute_answer_ranking():
    global answer_vector, word_to_rank
    global sim_top1, sim_top20, sim_top1000

    word_to_rank.clear()

    sims = []
    for w in words_list:
        v = np.array(embedding_dict[w])
        sims.append((w, cosine_sim(answer_vector, v)))

    sims.sort(key=lambda x: x[1], reverse=True)

    for idx, (w, _) in enumerate(sims):
        word_to_rank[w] = idx + 1

    sim_top1 = sims[0][1]
    sim_top20 = sims[19][1] if len(sims) >= 20 else sims[-1][1]
    sim_top1000 = sims[999][1] if len(sims) >= 1000 else sims[-1][1]

    print("[INFO] 정답 기준 랭킹 테이블 생성 완료")


# ============================================================
#  6) REQUEST MODELS
# ============================================================

class SetAnswerRequest(BaseModel):
    answer: str


class GuessRequest(BaseModel):
    team: str
    word: str


# ============================================================
#  7) ENDPOINT — 정답 설정
# ============================================================

@app.post("/set_answer")
def set_answer(req: SetAnswerRequest):
    global answer_word, answer_vector, submissions

    if req.answer not in embedding_dict:
        return JSONResponse({"error": "정답이 사전 단어(words_5000.json)에 없습니다."})

    answer_word = req.answer
    answer_vector = np.array(embedding_dict[answer_word])
    submissions = []

    compute_answer_ranking()

    return {"status": "ok", "answer": answer_word}


# ============================================================
#  8) ENDPOINT — 추측 입력
# ============================================================

@app.post("/guess")
def guess(req: GuessRequest):
    global submissions

    if answer_word is None:
        return JSONResponse({"error": "정답이 아직 설정되지 않았습니다."})

    team = req.team.strip()
    word = req.word.strip()

    if len(word) == 0:
        return JSONResponse({"error": "단어가 비어 있습니다."})

    # 중복 제출 방지 (같은 팀이 같은 단어 제출)
    for s in submissions:
        if s["team"] == team and s["word"] == word:
            return {"result": "duplicate"}

    # 정답 처리
    if word == answer_word:
        entry = {
            "team": team,
            "word": word,
            "is_answer": True,
            "rank": 1,
            "score": 100,
            "cosine": 1.0
        }
        submissions.append(entry)
        return {"result": "correct", "entry": entry}

    # 벡터 계산
    if word in embedding_dict:
        vec = np.array(embedding_dict[word])
        cos = cosine_sim(answer_vector, vec)
        score = score_from_cosine(cos)
        rank = word_to_rank.get(word, None)
    else:
        # 사전 외 단어: ??? 처리
        cos = cosine_sim(
            answer_vector,
            np.array([0] * len(answer_vector))
        )
        score = score_from_cosine(cos)
        rank = None

    entry = {
        "team": team,
        "word": word,
        "is_answer": False,
        "rank": rank if rank is not None else "???",
        "score": score,
        "cosine": cos
    }

    submissions.append(entry)
    return {"result": "ok", "entry": entry}


# ============================================================
#  9) ENDPOINT — 리더보드
# ============================================================

@app.get("/leaderboard")
def leaderboard():
    sorted_submissions = sorted(submissions, key=lambda x: (-x["score"]))
    return {
        "submissions": sorted_submissions,
        "answer": answer_word,
        "sim_top1": sim_top1,
        "sim_top20": sim_top20,
        "sim_top1000": sim_top1000
    }


# ============================================================
#  10) STATIC PAGES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/game", response_class=HTMLResponse)
def game_page():
    with open("static/game.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ============================================================
#  11) RUN (HF에서는 무시됨)
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
