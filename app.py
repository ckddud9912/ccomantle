import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

from fasttext_loader import get_vector  # 내부 구현은 sentence-transformers

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
    - 음수 → -20 고정
    - [0, 1] 범위 → 0 ~ 70 스케일링
    """
    if cosine < 0:
        return -20.0
    return round(cosine * 70.0, 2)


def compute_leaderboard_stats():
    """
    submissions 기준으로 1위, 10위, 1000위 점수 계산.
    없으면 None.
    """
    if not submissions:
        return None, None, None

    scores = [s["score"] for s in submissions]

    best = scores[0]
    tenth = scores[9] if len(scores) >= 10 else scores[-1]
    thousandth = scores[999] if len(scores) >= 1000 else scores[-1]

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
    """
    정답 단어를 설정하고, 그에 대한 벡터를 미리 계산.
    기존 제출 기록(submissions)은 초기화.
    """
    global answer_word, answer_vector, submissions

    answer = data.answer.strip()
    if not answer:
        return {"ok": False, "error": "정답 단어가 비어 있습니다."}

    vec = get_vector(answer)
    # vec가 None이어도 일단 정답으로 사용 가능. (다른 단어도 전부 -20으로 나올 수 있음)

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

    vec = get_vector(w)

    if vec is None or answer_vector is None:
        score = -20.0
    else:
        cos = cosine_similarity(vec, answer_vector)
        score = score_from_cosine(cos)

    # 정답 단어 직접 제출 시 70점 고정
    if w == answer_word:
        score = 70.0

    entry = {
        "team": t,
        "word": w,
        "score": score,
    }

    submissions.append(entry)
    submissions.sort(key=lambda x: x["score"], reverse=True)

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
