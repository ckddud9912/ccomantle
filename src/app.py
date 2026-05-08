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

EMBEDDING_FILE = os.path.join(DATA_DIR, "embedding_dictionary_e5.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[INFO] Loading embeddings from {EMBEDDING_FILE}...")
    store = load_store(EMBEDDING_FILE)
    print(f"[INFO] Loaded embeddings ({len(store)} words).")
    app.state.game = GameState(store=store)
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
