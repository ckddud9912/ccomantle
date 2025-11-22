from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# SBERT 로드
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

ANSWER = None
ANSWER_VEC = None
submissions = []


def cosine_sim(v1, v2):
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return dot / norm if norm != 0 else 0.0


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/game")
def game():
    return FileResponse("static/game.html")


@app.get("/admin")
def admin():
    return FileResponse("static/admin.html")


@app.post("/set_answer")
async def set_answer(word: str = Form(...)):
    global ANSWER, ANSWER_VEC, submissions

    word = word.strip()
    if not word:
        return JSONResponse({"ok": False, "error": "정답 단어를 입력하세요."})

    ANSWER = word
    ANSWER_VEC = model.encode([ANSWER], convert_to_numpy=True)[0]
    submissions = []

    return JSONResponse({"ok": True, "answer": ANSWER})


@app.get("/guess")
async def guess(word: str, team: str):
    global ANSWER, ANSWER_VEC, submissions
    if ANSWER is None:
        return {"ok": False, "error": "진행자가 아직 정답을 설정하지 않았습니다."}

    word = word.strip()
    team = team.strip()
    if not word:
        return {"ok": False, "error": "단어를 입력하세요."}

    if word == ANSWER:
        similarity = 100.0
    else:
        vec = model.encode([word], convert_to_numpy=True)[0]
        similarity = float(cosine_sim(vec, ANSWER_VEC) * 100)
        similarity = max(0.0, round(similarity, 2))

    submissions.append({"team": team, "word": word, "similarity": similarity})
    return {"ok": True, "similarity": similarity}


@app.get("/leaderboard")
async def leaderboard():
    sorted_list = sorted(submissions, key=lambda x: x["similarity"], reverse=True)
    return {"ok": True, "leaderboard": sorted_list}
