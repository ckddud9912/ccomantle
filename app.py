import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

from fasttext_loader import get_vector  # sentence-transformers 기반 구현

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
    코사인 유사도(-1~1)를 점수(-100~100)로 선형 매핑.
    """
    return round(cosine * 100.0, 2)


def compute_leaderboard_stats():
    """
    1위 / 20위 / 1000위 점수 계산.
    정답("정답!") 문자열은 제외하고 숫자 점수만 사용.
    """
    numeric_scores = [s["score"] for s in submissions if isinstance(s["score"], (int, float))]
    if not numeric_scores:
        return None, None, None

    best = numeric_scores[0]
    twentieth = numeric_scores[19] if len(numeric_scores) >= 20 else None
    thousandth = numeric_scores[999] if len(numeric_scores) >= 1000 else None

    return best, twentieth, thousandth


def recompute_ranks():
    """
    submissions 리스트의 현재 순서를 기준으로 rank 재계산.
    """
    for idx, s in enumerate(submissions, start=1):
        s["rank"] = idx


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

    # -------------------------
    #  정답 제출 → "정답!" 처리
    # -------------------------
    if w == answer_word:
        entry = {
            "team": t,
            "word": w,
            "score": "정답!",
            "is_answer": True,
        }
        submissions.append(entry)

        # 정답은 항상 맨 위로
        answers = [s for s in submissions if s.get("is_answer")]
        numeric = [s for s in submissions if not s.get("is_answer") and isinstance(s["score"], (int, float))]
        numeric.sort(key=lambda x: x["score"], reverse=True)
        strings = [s for s in submissions if not s.get("is_answer") and isinstance(s["score"], str)]

        submissions[:] = answers + numeric + strings
        recompute_ranks()

        return {
            "ok": True,
            "team": t,
            "word": w,
            "score": "정답!",
            "rank": entry["rank"],
        }

    # -------------------------
    #  일반 단어 처리
    # -------------------------
    vec = get_vector(w)

    if vec is None or answer_vector is None:
        score = -100.0
    else:
        cos = cosine_similarity(vec, answer_vector)
        score = score_from_cosine(cos)

    entry = {
        "team": t,
        "word": w,
        "score": score,
        "is_answer": False,
    }

    submissions.append(entry)

    # 정답 그룹, 숫자 그룹, 기타 문자열 그룹 순으로 정렬
    answers = [s for s in submissions if s.get("is_answer")]
    numeric = [s for s in submissions if not s.get("is_answer") and isinstance(s["score"], (int, float))]
    numeric.sort(key=lambda x: x["score"], reverse=True)
    strings = [s for s in submissions if not s.get("is_answer") and isinstance(s["score"], str)]

    submissions[:] = answers + numeric + strings
    recompute_ranks()

    return {
        "ok": True,
        "team": t,
        "word": w,
        "score": score,
        "rank": entry["rank"],
    }


# ===========================
#   API: LEADERBOARD
# ===========================
@app.get("/leaderboard")
def leaderboard():
    best, twentieth, thousandth = compute_leaderboard_stats()

    data = []
    for idx, s in enumerate(submissions, start=1):
        data.append({
            "team": s["team"],
            "word": s["word"],
            "score": s["score"],
            "rank": s.get("rank", idx),
            "is_answer": s.get("is_answer", False),
        })

    # 정답 단어의 자체 유사도 점수(항상 100점) – answer가 설정된 경우만 사용
    answer_score = 100.0 if answer_word is not None else None

    return {
        "ok": True,
        "answer_score": answer_score,
        "best": best,
        "twentieth": twentieth,
        "thousandth": thousandth,
        "leaderboard": data,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
