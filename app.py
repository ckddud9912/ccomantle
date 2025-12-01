import subprocess
import os
import run_embed
import uvicorn

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import json
from typing import List, Dict, Optional

app = FastAPI()

# ============================
# 50,000 단어 사전 자동 생성
# ============================
WORDS_50K = "words_50000.json"

if not os.path.exists(WORDS_50K):
    print("[자동 실행] words_50000.json 없음 → make_words.py 실행 시작")
    try:
        subprocess.run(["python", "make_words.py"], check=True)
        print("[자동 실행] make_words.py 실행 완료!")
    except Exception as e:
        print("[ERROR] make_words.py 실행 실패:", e)

# Static 파일
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================================================
# GLOBAL STATE
# ==================================================

answer_word: Optional[str] = None
answer_vector: Optional[np.ndarray] = None
submissions: List[Dict] = []

embedding_dict: Dict[str, List[float]] = {}
words_list: List[str] = []
word_to_rank: Dict[str, int] = {}

sim_top1 = None
sim_top20 = None
sim_top1000 = None


# ==================================================
# LOAD EMBEDDING DICTIONARY
# ==================================================

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


# ==================================================
# COSINE 유사도
# ==================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ==================================================
# 정답 설정 시 랭킹 테이블 생성
# ==================================================

def compute_answer_ranking():
    global answer_vector, word_to_rank
    global sim_top1, sim_top20, sim_top1000

    sims = []
    for w in words_list:
        v = np.array(embedding_dict[w])
        sims.append((w, cosine_sim(answer_vector, v)))

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
# ENDPOINT — 정답 설정
# ==================================================

@app.post("/set_answer")
def set_answer(req: SetAnswerRequest):
    global answer_word, answer_vector, submissions

    if req.answer not in embedding_dict:
        return JSONResponse({"error": "정답이 사전 단어에 없습니다."})

    answer_word = req.answer
    answer_vector = np.array(embedding_dict[answer_word])
    submissions = []

    compute_answer_ranking()

    return {"status": "ok", "answer": answer_word}


# ==================================================
# ENDPOINT — 추측 입력
# ==================================================

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

    # 정답 입력
    if word == answer_word:
        entry = {
            "team": team,
            "word": word,
            "is_answer": True,
            "rank": 1,
            "similarity": 1.000
        }
        submissions.append(entry)
        return {"result": "correct", "entry": entry}

    # 벡터 유사도 계산
    if word in embedding_dict:
        vec = np.array(embedding_dict[word])
        cos = cosine_sim(answer_vector, vec)
        similarity = round(float(cos), 3)
        rank = word_to_rank.get(word, None)
    else:
        similarity = -1.000
        rank = None

    entry = {
        "team": team,
        "word": word,
        "is_answer": False,
        "rank": rank if rank is not None else "???",
        "similarity": similarity
    }

    submissions.append(entry)
    return {"result": "ok", "entry": entry}


# ==================================================
# 리더보드
# ==================================================

@app.get("/leaderboard")
def leaderboard():
    sorted_submissions = sorted(submissions, key=lambda x: (-x["similarity"]))
    return {
        "submissions": sorted_submissions,
        "answer": answer_word,
        "sim_top1": sim_top1,
        "sim_top20": sim_top20,
        "sim_top1000": sim_top1000
    }


# ==================================================
# STATIC PAGES
# ==================================================

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


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
