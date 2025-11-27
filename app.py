import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Optional

from fasttext_loader import get_vector


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
def cosine(a, b):
    if a is None or b is None:
        return -1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------
#  원조 꼬맨틀 점수 공식 복원
# ---------------------------
def score_from_cosine(cos):
    """
    원조 꼬맨틀 분포에 맞춘 Nonlinear 스코어 변환.

    1) cosine -1~1 → similarity 0~1
    2) S-curve exponent ~5.8 로 상위권만 확 살림
    3) 0~1 → 0~100 변환
    4) 저점수는 강하게 눌러줌(원조 꼬맨틀 특징)
    5) -100 ~ 100 최종 스케일링
    """
    if cos is None:
        return -100.0

    sim = (cos + 1) / 2
    curved = sim ** 5.8
    score = curved * 100

    if score < 30:
        score = score * 0.35

    final = score * 2 - 100
    return round(final, 2)


def recompute_ranks():
    for i, s in enumerate(submissions, start=1):
        s["rank"] = i


def stats():
    numeric = [s["score"] for s in submissions if isinstance(s["score"], (int, float))]
    if not numeric:
        return None, None, None
    best = numeric[0]
    twentieth = numeric[19] if len(numeric) >= 20 else None
    thousandth = numeric[999] if len(numeric) >= 1000 else None
    return best, twentieth, thousandth


# ===========================
#   SCHEMAS
# ===========================
class AnswerRequest(BaseModel):
    answer: str


# ===========================
#   ROUTES - HTML
# ===========================
@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/game", response_class=HTMLResponse)
def game():
    with open("static/game.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/admin", response_class=HTMLResponse)
def admin():
    with open("static/admin.html", "r", encoding="utf-8") as f:
        return f.read()


# ===========================
#   API: SET ANSWER
# ===========================
@app.post("/set_answer")
def set_answer(body: AnswerRequest):
    global answer_word, answer_vector, submissions

    clean = body.answer.strip()
    if not clean:
        return {"ok": False, "error": "정답 단어가 비어 있습니다."}

    vec = get_vector(clean)

    answer_word = clean
    answer_vector = vec
    submissions = []

    return {"ok": True, "answer": clean}


# ===========================
#   API: GUESS
# ===========================
@app.get("/guess")
def guess(word: str, team: str = "팀"):
    global answer_word, answer_vector, submissions

    if answer_word is None:
        return {"ok": False, "error": "정답이 아직 설정되지 않았습니다."}

    word = word.strip()
    team = team.strip() or "팀"

    # 정답 처리
    if word == answer_word:
        entry = {
            "team": team,
            "word": word,
            "score": "정답!",
            "is_answer": True
        }
        submissions.append(entry)

        answers = [s for s in submissions if s.get("is_answer")]
        numeric = [s for s in submissions if not s.get("is_answer") and isinstance(s["score"], (int, float))]
        numeric.sort(key=lambda x: x["score"], reverse=True)
        strings = [s for s in submissions if not s.get("is_answer") and isinstance(s["score"], str)]

        submissions[:] = answers + numeric + strings
        recompute_ranks()

        return {"ok": True, "team": team, "word": word, "score": "정답!", "rank": entry["rank"]}

    vec = get_vector(word)
    cos = cosine(vec, answer_vector)
    score = score_from_cosine(cos)

    entry = {
        "team": team,
        "word": word,
        "score": score,
        "is_answer": False
    }
    submissions.append(entry)

    answers = [s for s in submissions if s.get("is_answer")]
    numeric = [s for s in submissions if not s.get("is_answer") and isinstance(s["score"], (int, float))]
    numeric.sort(key=lambda x: x["score"], reverse=True)
    strings = [s for s in submissions if not s.get("is_answer") and isinstance(s["score"], str)]

    submissions[:] = answers + numeric + strings
    recompute_ranks()

    return {"ok": True, "team": team, "word": word, "score": score, "rank": entry["rank"]}


# ===========================
#   API: LEADERBOARD
# ===========================
@app.get("/leaderboard")
def leaderboard():
    best, twentieth, thousandth = stats()

    out = []
    for s in submissions:
        out.append({
            "team": s["team"],
            "word": s["word"],
            "score": s["score"],
            "rank": s["rank"],
            "is_answer": s.get("is_answer", False)
        })

    answer_score = 100 if answer_word else None

    return {
        "ok": True,
        "answer_score": answer_score,
        "best": best,
        "twentieth": twentieth,
        "thousandth": thousandth,
        "leaderboard": out
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
