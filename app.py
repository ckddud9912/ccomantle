import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Dict, Optional
import numpy as np
from pydantic import BaseModel
from fasttext_loader import get_vector

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# ===============================
#   GLOBAL GAME STATE
# ===============================
answer_word: Optional[str] = None
answer_vector: Optional[np.ndarray] = None

submissions: List[Dict] = []


# ===============================
#   UTILS
# ===============================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return -1.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_from_cosine(cosine: float) -> float:
    if cosine < 0:
        return -20.0
    return round(cosine * 70.0, 2)


# ===============================
#   ROUTES
# ===============================
@app.get("/", response_class=HTMLResponse)
def main_page():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/game", response_class=HTMLResponse)
def game_page():
    with open("static/game.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return f.read()


class Answer(BaseModel):
    answer: str


@app.post("/set_answer")
def set_answer(data: Answer):
    global answer_word, answer_vector, submissions

    answer_word = data.answer.strip()
    answer_vector = get_vector(answer_word)
    submissions = []

    return {"ok": True, "answer": answer_word}


@app.get("/guess")
def guess(word: str, team: str = "팀"):
    global answer_word, answer_vector, submissions

    word = word.strip()
    vec = get_vector(word)

    if vec is None:
        score = -20
    else:
        cos = cosine_similarity(vec, answer_vector)
        score = score_from_cosine(cos)

    if word == answer_word:
        score = 70.0

    entry = {
        "team": team,
        "word": word,
        "score": score
    }
    submissions.append(entry)

    submissions.sort(key=lambda x: x["score"], reverse=True)

    rank = submissions.index(entry) + 1

    return {
        "ok": True,
        "team": team,
        "word": word,
        "score": score,
        "rank": rank
    }


@app.get("/leaderboard")
def leaderboard():
    if len(submissions) == 0:
        return {
            "ok": True,
            "best": None,
            "tenth": None,
            "thousandth": None,
            "leaderboard": []
        }

    scores = [s["score"] for s in submissions]

    best = scores[0]
    tenth = scores[9] if len(scores) >= 10 else scores[-1]
    thousandth = scores[999] if len(scores) >= 1000 else scores[-1]

    return {
        "ok": True,
        "best": best,
        "tenth": tenth,
        "thousandth": thousandth,
        "leaderboard": submissions
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, lifespan="off")
