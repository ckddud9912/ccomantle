import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

from fasttext_loader import get_vector  # 내부는 sentence-transformers 기반

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# ===========================
#   GAME STATE
# ===========================
answer_word: Optional[str] = None
answer_vector: Optional[np.ndarray] = None
submissions: List[Dict] = []


# ===========================
#   UTILS
# ===========================
def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def score_from_cosine(cosine: float) -> float:
    """
    코사인 유사도를 점수로 변환.
    - 음수 → -10
    - 0~1 → 0~100 선형 스케일
    """
    if cosine < 0:
        return -10.0
    return round(cosine * 100.0, 2)


def compute_leaderboard_stats():
    if not submissions:
        return None, None, None

    numeric_scores = [s["score"] for s in submissions if isinstance(s["score"], (int, float))]

    if not numeric_scores:
        return None, None, None

    best = numeric_scores[0]
    tenth = numeric_scores[9] if len(numeric_scores) >= 10 else numeric_scores[-1]
    thousandth = numeric_scores[999] if len(numeric_scores) >= 1000 else numeric_scores[-1]

    return best, tenth, thousandth


# ===========================
#   SCHEMA
# ===========================
class AnswerRequest(BaseModel):
    answer: str


# ===========================
#   HTML ROUTES
# ===========================
@app.get("/", response_class=HTMLResponse)
def main_page():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/game", response_class=HTMLResponse)
def game_page():
    with open("static/game.html", encoding="utf-8") as f:
        return f.read()


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    with open("static/admin.html", encoding="utf-8") as f:
        return f.read()


# ===========================
#   API: SET ANSWER
# ===========================
@app.post("/set_answer")
def set_answer(data: AnswerRequest):
    global answer_word, answer_vector, submissions

    answer = data.answer.strip()
    if not answer:
        return {"ok": False, "error": "정답 단어가 비어 있습니다."}

    vec = get_vector(answer)

    answer_word = answer
    answer_vector = vec
    submissions = []

    return {"ok": True, "answer": answer}


# ===========================
#   API: GUESS WORD
# ===========================
@app.get("/guess")
def guess(word: str, team: str = "팀"):
    global answer_word, answer_vector, submissions

    if answer_word is None:
        return {"ok": False, "error": "아직 정답이 설정되지 않았습니다."}

    w = word.strip()
    t = team.strip() or "팀"

    # 정답 직접 제출 → 숫자 점수 대신 "정답!"
    if w == answer_word:
        entry = {
            "team": t,
            "word": w,
            "score": "정답!",
            "rank": 1
        }
        submissions.insert(0, entry)
        return {
            "ok": True,
            "team": t,
            "word": w,
            "score": "정답!",
            "rank": 1
        }

    # 일반 단어 처리
    vec = get_vector(w)

    if vec is None or answer_vector is None:
        score = -10.0
    else:
        cos = cosine_similarity(vec, answer_vector)
        score = score_from_cosine(cos)

    entry = {
        "team": t,
        "word": w,
        "score": score,
    }

    submissions.append(entry)

    numeric_first = [s for s in submissions if isinstance(s["score"], (int, float))]
    numeric_first.sort(key=lambda x: x["score"], reverse=True)

    others = [s for s in submissions if isinstance(s["score"], str)]

    submissions[:] = numeric_first + others

    rank = submissions.index(entry) + 1
    entry["rank"] = rank

    return {
        "ok": True,
        "team": t,
        "word": w,
        "score": score,
        "rank": rank,
    }


# ===========================
#   API: LEADERBOARD
# ===========================
@app.get("/leaderboard")
def leaderboard():
    best, tenth, thousandth = compute_leaderboard_stats()

    data = []
    for idx, s in enumerate(submissions, start=1):
        data.append({
            "team": s["team"],
            "word": s["word"],
            "score": s["score"],
            "rank": s.get("rank", idx),
        })

    return {
        "ok": True,
        "best": best,
        "tenth": tenth,
        "thousandth": thousandth,
        "leaderboard": data,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
