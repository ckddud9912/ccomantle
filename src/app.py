# === app.py - 완성본 (유사도 정규화 포함) ===
import os
import math
from typing import List, Dict, Optional

import numpy as np
import orjson
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ==================================================
# 전역 상태
# ==================================================
embedding_dict: Optional[Dict[str, List[float]]] = None
word_to_rank: Dict[str, int] = {}

answer_word: Optional[str] = None
answer_vector: Optional[np.ndarray] = None

MAX_ROUNDS = 6
current_round: int = 1
rounds: Dict[int, List[Dict]] = {i: [] for i in range(1, MAX_ROUNDS + 1)}

game_finished: bool = False

sim_top1: Optional[float] = None
sim_top20: Optional[float] = None
sim_top1000: Optional[float] = None

# 유사도 스케일링 파라미터
SIM_ALPHA: float = 1.0          # s' = s ** SIM_ALPHA
TARGET_TOP1000: float = 0.63    # 1000위 목표 값


# ==================================================
# 디렉토리
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(ROOT_DIR, "static")
DATA_DIR = os.path.join(ROOT_DIR, "data")


# ==================================================
# FastAPI
# ==================================================
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==================================================
# 임베딩 로드
# ==================================================
@app.on_event("startup")
async def load_embeddings() -> None:
    global embedding_dict

    emb_path = os.path.join(DATA_DIR, "embedding_dictionary_e5.json")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"{emb_path} not found")

    with open(emb_path, "rb") as f:
        print(f"[INFO] Loading embeddings from {emb_path}...")
        embedding_dict = orjson.loads(f.read())

    print(f"[INFO] Loaded embeddings ({len(embedding_dict)} words).")


# ==================================================
# 유틸
# ==================================================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def apply_sim_scaling(raw_sim: float) -> float:
    """
    raw_sim: 코사인 유사도 [-1,1] 가정.
    음수는 0으로 클램프하고, 양수는 s ** SIM_ALPHA 로 변환.
    """
    if raw_sim <= 0:
        return 0.0
    return raw_sim ** SIM_ALPHA


def compute_answer_ranking() -> None:
    """
    1) 정답 기준 raw similarity 계산
    2) raw 기준으로 랭킹/1000위 값 계산
    3) 1000위가 TARGET_TOP1000 이 되도록 SIM_ALPHA 계산
    4) 스케일링 된 값 기준으로 sim_top1,20,1000 갱신
    """
    global word_to_rank, sim_top1, sim_top20, sim_top1000, SIM_ALPHA

    sims_raw = []
    for w, vec_list in embedding_dict.items():
        v = np.array(vec_list, dtype=float)
        sims_raw.append((w, cosine_sim(answer_vector, v)))

    sims_raw.sort(key=lambda x: x[1], reverse=True)
    word_to_rank = {w: idx + 1 for idx, (w, _) in enumerate(sims_raw)}

    # raw 기준 상위 값들
    sim_top1_raw = sims_raw[0][1]
    sim_top20_raw = sims_raw[min(19, len(sims_raw) - 1)][1]
    sim_top1000_raw = sims_raw[min(999, len(sims_raw) - 1)][1]

    # SIM_ALPHA 계산: (sim_top1000_raw ** alpha = TARGET_TOP1000)
    if 0 < sim_top1000_raw < 1:
        SIM_ALPHA = math.log(TARGET_TOP1000) / math.log(sim_top1000_raw)
    else:
        SIM_ALPHA = 1.0

    print(f"[INFO] sim_top1000_raw={sim_top1000_raw:.4f}, "
          f"SIM_ALPHA={SIM_ALPHA:.4f}")

    # 스케일링 된 값으로 최종 지표 셋팅
    sim_top1 = apply_sim_scaling(sim_top1_raw)
    sim_top20 = apply_sim_scaling(sim_top20_raw)
    sim_top1000 = apply_sim_scaling(sim_top1000_raw)


# ==================================================
# Request Models
# ==================================================
class SetAnswerRequest(BaseModel):
    answer: str


class GuessRequest(BaseModel):
    team: str
    word: str


class RoundRequest(BaseModel):
    round: int


# ==================================================
# API: 정답 설정
# ==================================================
@app.post("/set_answer")
async def set_answer(req: SetAnswerRequest):
    global answer_word, answer_vector, rounds, current_round, game_finished

    if req.answer not in embedding_dict:
        return JSONResponse({"error": "사전에 없는 단어입니다."})

    answer_word = req.answer
    answer_vector = np.array(embedding_dict[answer_word], dtype=float)

    rounds = {i: [] for i in range(1, MAX_ROUNDS + 1)}
    current_round = 1
    game_finished = False

    compute_answer_ranking()
    return {"status": "ok", "answer": answer_word}


# ==================================================
# API: 라운드 변경
# ==================================================
@app.post("/set_round")
async def set_round(req: RoundRequest):
    global current_round
    if 1 <= req.round <= MAX_ROUNDS:
        current_round = req.round
        return {"status": "ok", "current_round": current_round}
    return JSONResponse({"error": "Invalid round"}, status_code=400)


# ==================================================
# API: 추측 제출
# ==================================================
@app.post("/guess")
async def guess(req: GuessRequest):
    global rounds

    if game_finished:
        return JSONResponse({"error": "경기가 종료되었습니다."})

    if answer_word is None:
        return JSONResponse({"error": "정답이 아직 설정되지 않았습니다."})

    team = req.team.strip()
    word = req.word.strip()

    for s in rounds[current_round]:
        if s["team"] == team:
            return {"result": "duplicate"}

    if word == answer_word:
        entry = {
            "round": current_round, "team": team, "word": word,
            "is_answer": True, "rank": 1, "similarity": 1.0,
        }
        rounds[current_round].append(entry)
        return {"result": "correct", "entry": entry}

    if word not in embedding_dict:
        return JSONResponse({"error": "사전에 없는 단어입니다."})

    vec = np.array(embedding_dict[word], dtype=float)
    raw_sim = cosine_sim(answer_vector, vec)
    norm_sim = apply_sim_scaling(raw_sim)

    similarity = round(norm_sim, 3)
    rank = word_to_rank.get(word)

    entry = {
        "round": current_round, "team": team, "word": word,
        "is_answer": False, "rank": rank, "similarity": similarity,
    }
    rounds[current_round].append(entry)

    return {"result": "ok", "entry": entry}


# ==================================================
# API: 리더보드
# ==================================================
@app.get("/leaderboard")
async def leaderboard():
    sorted_rounds = {
        str(r): sorted(rounds[r], key=lambda x: x["similarity"], reverse=True)
        for r in rounds
    }

    return {
        "current_round": current_round,
        "max_rounds": MAX_ROUNDS,
        "answer": answer_word,
        "sim_top1": sim_top1,
        "sim_top20": sim_top20,
        "sim_top1000": sim_top1000,
        "rounds": sorted_rounds,
        "finished": game_finished
    }


# ==================================================
# API: top1000
# ==================================================
@app.get("/top1000")
async def top1000():
    sims = []
    for w, vec_list in embedding_dict.items():
        v = np.array(vec_list, dtype=float)
        raw_sim = cosine_sim(answer_vector, v)
        norm_sim = apply_sim_scaling(raw_sim)
        sims.append((w, norm_sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:1000]

    return {
        "answer": answer_word,
        "top1000": [
            {"rank": i + 1, "word": w, "similarity": round(sim, 4)}
            for i, (w, sim) in enumerate(top)
        ]
    }


# ==================================================
# API: 경기 종료
# ==================================================
@app.post("/end_game")
async def end_game():
    global game_finished
    game_finished = True
    return {"status": "finished"}


# ==================================================
# API: 최종 성적 계산
# ==================================================
@app.get("/final_result")
async def final_result():
    team_scores = {}

    for r in rounds.values():
        for s in r:
            if s["team"] not in team_scores:
                team_scores[s["team"]] = []
            if isinstance(s["similarity"], float):
                team_scores[s["team"]].append(s["similarity"])

    final = []
    for t, sims in team_scores.items():
        avg = sum(sims) / len(sims)
        final.append({"team": t, "avg": round(avg, 4)})

    final.sort(key=lambda x: x["avg"], reverse=True)
    return {"result": final}


# ==================================================
# HTML
# ==================================================
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read())


@app.get("/game", response_class=HTMLResponse)
async def game():
    return HTMLResponse(open(os.path.join(STATIC_DIR, "game.html"), encoding="utf-8").read())


@app.get("/admin", response_class=HTMLResponse)
async def admin():
    return HTMLResponse(open(os.path.join(STATIC_DIR, "admin.html"), encoding="utf-8").read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
