from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- SBERT 모델 로드 -----
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

ANSWER = None
ANSWER_VEC = None
submissions = []

# 전체 단어 분포용
VOCAB_TOKENS = []
VOCAB_EMBEDS = None
VOCAB_SIMS_SORTED = None  # 정답 기준 전체 유사도 분포 (내림차순)
SIM_STATS = {"top1": None, "top10": None, "top1000": None}


def convert_similarity(cos_sim: float) -> float:
    """
    cos_sim(-1~1)을 -20 ~ +70 범위로 스케일
    """
    return round(((cos_sim + 1) / 2) * 90 - 20, 2)


def cosine_sim(v1, v2) -> float:
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm == 0:
        return 0.0
    return dot / norm


def build_vocab():
    """
    SBERT 토크나이저에서 한국어 토큰들만 추려서 vocab 구성 후 임베딩 계산
    너무 많으면 상위 N개만 씀.
    """
    global VOCAB_TOKENS, VOCAB_EMBEDS

    tokenizer = model.tokenizer
    vocab = list(tokenizer.get_vocab().keys())

    def is_korean(s: str) -> bool:
        return any("가" <= ch <= "힣" for ch in s)

    # ## 붙은 서브워드는 빼고, 한글 포함 토큰만
    tokens = [t for t in vocab if is_korean(t) and not t.startswith("##")]

    # 너무 많으면 앞에서 N개만 사용 (속도/메모리 타협)
    MAX_VOCAB = 20000
    VOCAB_TOKENS = tokens[:MAX_VOCAB]

    print(f"📚 Vocab tokens: {len(VOCAB_TOKENS)}개 선택")

    VOCAB_EMBEDS = model.encode(VOCAB_TOKENS, convert_to_numpy=True)
    print("✅ Vocab 임베딩 완료")


def build_answer_distribution():
    """
    정답 단어 기준으로 vocab 전체 유사도 분포 계산 + top1/top10/top1000 저장
    """
    global VOCAB_SIMS_SORTED, SIM_STATS

    if ANSWER_VEC is None or VOCAB_EMBEDS is None:
        return

    # 정답 벡터 정규화
    ans_vec = ANSWER_VEC / np.linalg.norm(ANSWER_VEC)

    # vocab 임베딩 정규화
    vocab_norm = VOCAB_EMBEDS / np.linalg.norm(VOCAB_EMBEDS, axis=1, keepdims=True)

    # cos 유사도
    sims = vocab_norm @ ans_vec  # shape: (V,)

    # 내림차순 정렬
    VOCAB_SIMS_SORTED = np.sort(sims)[::-1]

    # 통계값 (스케일 변환 후)
    if len(VOCAB_SIMS_SORTED) > 0:
        top1 = convert_similarity(VOCAB_SIMS_SORTED[0])
        top10 = convert_similarity(VOCAB_SIMS_SORTED[min(9, len(VOCAB_SIMS_SORTED) - 1)])
        top1000 = convert_similarity(VOCAB_SIMS_SORTED[min(999, len(VOCAB_SIMS_SORTED) - 1)])

        SIM_STATS = {
            "top1": top1,
            "top10": top10,
            "top1000": top1000
        }
        print("📊 유사도 통계:", SIM_STATS)


def get_rank_from_distribution(cos_sim: float) -> int:
    """
    정답 기준 전체 vocab 유사도 분포(VOCAB_SIMS_SORTED)에서
    cos_sim이 상위 몇 등에 해당하는지 계산.
    (내림차순 배열이므로 -cos_sim 기준으로 searchsorted)
    """
    if VOCAB_SIMS_SORTED is None:
        return 1000000

    # 내림차순이니까 음수로 바꿔서 이분 탐색
    idx = int(np.searchsorted(-VOCAB_SIMS_SORTED, -cos_sim, side="left"))
    return idx + 1  # 등수는 1부터 시작


# ----- 스타트업에서 vocab 준비 -----
@app.on_event("startup")
def on_startup():
    print("🚀 서버 시작 - vocab 구축 중...")
    build_vocab()
    print("🚀 준비 완료")


# ----- 페이지 라우팅 -----
@app.get("/")
def page_index():
    return FileResponse("static/index.html")


@app.get("/game")
def page_game():
    return FileResponse("static/game.html")


@app.get("/admin")
def page_admin():
    return FileResponse("static/admin.html")


# ----- 진행자: 정답 설정 -----
@app.post("/set_answer")
def set_answer(word: str = Form(...)):
    global ANSWER, ANSWER_VEC, submissions

    word = word.strip()
    if not word:
        return JSONResponse({"ok": False, "error": "정답 단어를 입력하세요."})

    ANSWER = word
    ANSWER_VEC = model.encode([ANSWER], convert_to_numpy=True)[0]

    submissions = []  # 리셋
    build_answer_distribution()  # 정답 기준 분포 갱신

    return JSONResponse({"ok": True, "answer": ANSWER, "stats": SIM_STATS})


# ----- 팀: 단어 제출 -----
@app.get("/guess")
def guess(word: str, team: str):
    global ANSWER, ANSWER_VEC

    if ANSWER is None or ANSWER_VEC is None:
        return {"ok": False, "error": "아직 진행자가 정답을 설정하지 않았습니다."}

    word = word.strip()
    team = (team or "").strip() or "이름없는 팀"

    if not word:
        return {"ok": False, "error": "단어를 입력하세요."}

    # SBERT 유사도 계산
    vec = model.encode([word], convert_to_numpy=True)[0]
    cos = cosine_sim(vec, ANSWER_VEC)

    similarity = convert_similarity(cos)  # -20 ~ +70 스케일
    submissions.append({
        "team": team,
        "word": word,
        "similarity": similarity,
        "cos": cos
    })

    return {
        "ok": True,
        "similarity": similarity
    }


# ----- 리더보드 -----
@app.get("/leaderboard")
def leaderboard():
    if ANSWER is None:
        return {"ok": False, "error": "정답이 아직 설정되지 않았습니다."}

    # 각 제출에 대해 유사도 순위 추정
    rows = []
    for s in submissions:
        rank = get_rank_from_distribution(s["cos"])
        display_rank = rank if rank <= 999 else "1000위 이상"

        rows.append({
            "team": s["team"],
            "word": s["word"],
            "similarity": s["similarity"],
            "rank": display_rank
        })

    # 유사도 점수 기준으로 정렬 (게임용)
    rows_sorted = sorted(rows, key=lambda x: x["similarity"], reverse=True)

    return {
        "ok": True,
        "leaderboard": rows_sorted,
        "stats": SIM_STATS
    }
