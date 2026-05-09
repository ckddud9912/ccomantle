import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from core.embeddings import load_store
from core.game import GameState


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(ROOT_DIR, "static")
DATA_DIR = os.path.join(ROOT_DIR, "data")

EMBEDDING_FILE = os.environ.get(
    "EMBEDDING_FILE",
    os.path.join(DATA_DIR, "embedding_dictionary_e5.json"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    임베딩 파일이 없어도 서버 자체는 기동된다(/health 가 503으로 알림).
    이렇게 해야 운영자가 컨테이너 로그를 보고 원인을 파악할 수 있고,
    배포 직후 임베딩 파일만 따로 올려도 재시작 한 번이면 복구 가능.
    """
    try:
        print(f"[INFO] Loading embeddings from {EMBEDDING_FILE}...")
        store = load_store(EMBEDDING_FILE)
        app.state.game = GameState(store=store)
        print(f"[INFO] Loaded embeddings ({len(store)} words). Ready.")
    except FileNotFoundError:
        print(f"[ERROR] Embedding file not found: {EMBEDDING_FILE}")
        print("[ERROR] Server will start but all game routes return 503.")
        print("[ERROR] Place the embedding JSON at the path above and restart.")
        app.state.game = None
    except Exception as e:
        print(f"[ERROR] Failed to load embeddings: {e}")
        app.state.game = None

    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(router)


@app.get("/")
async def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/game")
async def game_page():
    return FileResponse(os.path.join(STATIC_DIR, "game.html"))


@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
