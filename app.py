from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from fasttext_loader import (
    load_fasttext,
    has_word,
    get_vector,
    cosine_sim,
    convert_similarity,
    calculate_ranking,
    stats_for_answer,
)

app = FastAPI()

# CORS 허용 (브라우저에서 접근 용이하게)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANSWER: str | None = None
ANSWER_RANKING: dict[str, int] | None = None
submissions: list[dict] = []


@app.on_event("startup")
def startup_event():
    # 서버 시작 시 모델 미리 로딩 (첫 실행은 오래 걸릴 수 있음)
    load_fasttext()
    print("🚀 서버 시작 완료")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/game")
def game_page():
    return FileResponse("static/game.html")


@app.get("/admin")
def admin_page():
    return FileResponse("static/admin.html")


@app.post("/set_answer")
def set_answer(word: str = Form(...)):
    """
    진행자가 정답 단어를 설정하는 엔드포인트.
    """
    global ANSWER, ANSWER_RANKING, submissions

    load_fasttext()

    word = word.strip()
    if not word:
        return JSONResponse({"ok": False, "error": "정답 단어를 입력하세요."})

    if not has_word(word):
        return JSONResponse({"ok": False, "error": "FastText 사전에 없는 단어입니다."})

    ANSWER = word
    submissions = []  # 라운드 초기화

    # 정답 기준 전체 순위 계산 (시간 다소 소요)
    sims, ranking = calculate_ranking(ANSWER)
    ANSWER_RANKING = ranking

    return {"ok": True, "answer": ANSWER}


@app.get("/guess")
def guess(word: str, team: str):
    """
    참가자(팀)가 단어를 제출하는 엔드포인트.
    """
    global ANSWER, ANSWER_RANKING, submissions

    if ANSWER is None:
        return {"ok": False, "error": "진행자가 아직 정답을 설정하지 않았습니다."}

    word = word.strip()
    team = (team or "").strip() or "이름없는 팀"

    if not word:
        return {"ok": False, "error": "단어를 입력하세요."}

    if not has_word(word):
        return {"ok": False, "error": "FastText 사전에 없는 단어입니다."}

    vec = get_vector(word)
    answer_vec = get_vector(ANSWER)

    raw_sim = cosine_sim(vec, answer_vec)
    similarity = convert_similarity(raw_sim)

    rank = None
    rank_label = "순위 미측정"
    if ANSWER_RANKING is not None:
        rank = ANSWER_RANKING.get(word)
        if rank is None or rank > 1000:
            rank_label = "1000위 이상"
        else:
            rank_label = f"{rank}위"

    submissions.append(
        {
            "team": team,
            "word": word,
            "similarity": similarity,
            "rank": rank,
            "rank_label": rank_label,
        }
    )

    return {
        "ok": True,
        "similarity": similarity,
        "rank": rank,
        "rank_label": rank_label,
    }


@app.get("/leaderboard")
def leaderboard():
    """
    제출된 모든 단어를 유사도 높은 순으로 정렬해서 리턴.
    """
    sorted_list = sorted(submissions, key=lambda x: x["similarity"], reverse=True)
    return {"ok": True, "leaderboard": sorted_list}


@app.get("/stats")
def stats():
    """
    원조 꼬맨틀 스타일 유사도 안내:
      - 가장 유사한 단어
      - 10번째 유사한 단어
      - 1,000번째 유사한 단어
    """
    if ANSWER is None:
        return {"ok": False}

    max_sim, top10_sim, top1000_sim = stats_for_answer(ANSWER)

    return {
        "ok": True,
        "max_sim": max_sim,
        "top10_sim": top10_sim,
        "top1000_sim": top1000_sim,
    }
