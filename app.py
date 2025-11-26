import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import numpy as np
from fasttext_loader import get_vector

# ============================================================
#  FastAPI 초기화 및 정적 파일 설정
# ============================================================
app = FastAPI()

# /static 경로로 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
#  전역 게임 상태 (메모리 기반)
#  - 한 Spaces 인스턴스 안에서만 유지됨
# ============================================================
answer_word: Optional[str] = None          # 진행자가 설정한 정답 단어(문자열)
answer_vector: Optional[np.ndarray] = None # 정답 단어의 FastText 벡터

# submissions: 단어 제출 기록
#   {
#       "team": 팀 이름,
#       "word": 제출 단어,
#       "score": 점수(-20 ~ 70),
#       "rank": 제출 시점 기준 순위(1등 = 1),
#   }
submissions: List[Dict] = []


# ============================================================
#  유틸 함수: 코사인 유사도 & 점수 스케일링
# ============================================================
def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """두 벡터의 코사인 유사도 계산. 벡터가 없으면 -1.0 반환."""
    if a is None or b is None:
        return -1.0
    if a.shape != b.shape:
        return -1.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def score_from_cosine(cosine: float) -> float:
    """
    코사인 유사도를 게임 점수 스케일로 변환.
    - 기본 스케일: -20 ~ 70
    - cosine < 0 이면 그냥 -20 고정
    """
    if cosine < 0:
        return -20.0
    # 0 ~ 1 구간을 0 ~ 70으로 선형 스케일링 후 반올림
    score = cosine * 70.0
    return round(float(score), 2)


def compute_leaderboard_stats() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    전체 submissions 를 기준으로
    - best (1위 점수)
    - tenth (10위 점수)
    - thousandth (1000위 점수)
    를 계산해서 반환.
    제출이 없으면 모두 None.
    """
    if not submissions:
        return None, None, None

    # submissions 는 항상 score 내림차순 정렬 상태를 유지하도록 관리
    scores = [s["score"] for s in submissions]

    best = scores[0]                             # 1위
    tenth = scores[9] if len(scores) >= 10 else scores[-1]
    thousandth = scores[999] if len(scores) >= 1000 else scores[-1]

    return best, tenth, thousandth


# ============================================================
#  Pydantic 모델
# ============================================================
class AnswerRequest(BaseModel):
    answer: str


# ============================================================
#  라우터: 메인 / 게임 / 관리자 HTML
# ============================================================
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


# ============================================================
#  API: 정답 설정
# ============================================================
@app.post("/set_answer")
def set_answer(data: AnswerRequest):
    """
    진행자가 정답 단어를 설정.
    - answer_word / answer_vector 갱신
    - 기존 submissions 초기화
    """
    global answer_word, answer_vector, submissions

    raw_answer = data.answer.strip()
    if not raw_answer:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "정답 단어가 비어 있습니다."}
        )

    # FastText 벡터 가져오기 (지연 로딩 + OOV 대응은 fasttext_loader 안에서 처리)
    vec = get_vector(raw_answer)

    answer_word = raw_answer
    answer_vector = vec
    submissions = []  # 정답 바뀌면 기존 기록 초기화

    return {"ok": True, "answer": answer_word}


# ============================================================
#  API: 단어 제출
#  - /guess?word=단어&team=팀명
# ============================================================
@app.get("/guess")
def guess(word: str, team: str = "팀"):
    """
    플레이어(팀)가 단어를 제출하는 엔드포인트.

    - 정답이 아직 설정되지 않았다면 에러 반환
    - 제출 단어 FastText 벡터 가져오기
      - 벡터 없음(OOV) → score = -20
      - 있으면 코사인 유사도 → -20 ~ 70 점수 스케일링
      - word == answer_word 이면 무조건 70점
    - submissions 에 기록 후 score 기준 내림차순 정렬
    - 해당 제출의 현재 순위를 계산해서 함께 반환
    """
    global answer_word, answer_vector, submissions

    if answer_word is None:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "아직 진행자가 정답을 설정하지 않았습니다."}
        )

    clean_word = word.strip()
    clean_team = team.strip() if team.strip() else "팀"

    vec = get_vector(clean_word)

    # 벡터가 없으면(OOV) -20점
    if vec is None or answer_vector is None:
        score = -20.0
    else:
        cos = cosine_similarity(vec, answer_vector)
        score = score_from_cosine(cos)

    # 정답 단어는 고정 70점
    if clean_word == answer_word:
        score = 70.0

    # 제출 기록 생성
    entry = {
        "team": clean_team,
        "word": clean_word,
        "score": score,
    }
    submissions.append(entry)

    # 점수 기준 내림차순 정렬
    submissions.sort(key=lambda x: x["score"], reverse=True)

    # 현재 제출의 순위 계산 (동점일 경우 먼저 들어온 순서를 우선시)
    # (same dict identity 사용 가능하지만 여기서는 값 비교)
    rank = 1
    for idx, s in enumerate(submissions, start=1):
        if s is entry:
            rank = idx
            break

    entry["rank"] = rank

    # 리더보드 메타 정보도 즉시 계산해서 내려줄 수도 있지만,
    # 이 엔드포인트에서는 제출 결과에 집중
    return {
        "ok": True,
        "team": clean_team,
        "word": clean_word,
        "score": score,
        "rank": rank,
    }


# ============================================================
#  API: 리더보드 조회
#  - /leaderboard
#  - best / tenth / thousandth 포함
# ============================================================
@app.get("/leaderboard")
def leaderboard():
    """
    전체 리더보드 및 유사도 기준 안내용 스탯을 내려주는 엔드포인트.

    응답 예시:
    {
      "ok": true,
      "best": 65.23,
      "tenth": 41.11,
      "thousandth": 12.33,
      "leaderboard": [
        {"team": "...", "word": "...", "score": 65.23, "rank": 1},
        ...
      ]
    }
    """
    best, tenth, thousandth = compute_leaderboard_stats()

    # submissions 내부에는 rank 키가 없을 수도 있으니 안전하게 처리
    # front에 내려줄 때는 rank가 항상 포함되도록 가공
    ranked_list: List[Dict] = []
    for idx, s in enumerate(submissions, start=1):
        ranked_list.append(
            {
                "team": s.get("team", ""),
                "word": s.get("word", ""),
                "score": s.get("score", -20.0),
                "rank": s.get("rank", idx),
            }
        )

    return {
        "ok": True,
        "best": best,
        "tenth": tenth,
        "thousandth": thousandth,
        "leaderboard": ranked_list,
    }


# ============================================================
#  로컬 실행용 (Hugging Face Spaces에서도 그대로 사용 가능)
#  lifespan="off" 로 HF health check 관련 CancelledError 완화
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        lifespan="off",
    )
