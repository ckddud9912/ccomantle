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

# .env 자동 로드 — 도커 컴포즈는 자체적으로 .env를 환경변수로 주입하지만,
# 직접 `python src/app.py` 로 실행할 때는 안 읽힘. python-dotenv 로 양쪽 통일.
# 이미 export 되어 있는 환경변수가 있으면 .env 값으로 덮어쓰지 않음 (override=False 기본).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
except ImportError:
    pass

# 빈 문자열도 "미설정"으로 간주. .env 파일에 EMBEDDING_FILE= 라고만
# 적혀 있으면 os.environ.get(..., default)가 빈 문자열을 반환하기 때문에
# `or` 로 fallback 처리해야 한다.
EMBEDDING_FILE = (
    os.environ.get("EMBEDDING_FILE")
    or os.path.join(DATA_DIR, "embedding_dictionary_e5.json")
)


def _try_hf_download() -> bool:
    """
    EMBEDDING_HF_REPO 환경변수가 설정되어 있으면 HuggingFace Hub에서 임베딩 파일을 다운로드.
    이미 EMBEDDING_FILE 위치에 파일이 있으면 호출되지 않음 (lifespan에서 분기).
    실패하면 False 반환 — 로컬 파일 fallback 또는 503 모드로 진입.
    """
    repo = os.environ.get("EMBEDDING_HF_REPO")
    if not repo:
        return False

    # 빈 문자열도 미설정으로 간주 → 기본값 사용
    filename = os.environ.get("EMBEDDING_HF_FILE") or "embedding_dictionary_e5.json"
    repo_type = os.environ.get("EMBEDDING_HF_TYPE") or "dataset"

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[WARN] huggingface_hub 미설치. pip install huggingface_hub 필요.")
        return False

    try:
        print(f"[INFO] HF Hub에서 다운로드: {repo}/{filename} ({repo_type})")
        os.makedirs(DATA_DIR, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type=repo_type,
            local_dir=DATA_DIR,
        )
        # local_dir 사용 시 downloaded 가 EMBEDDING_FILE 와 다른 경로일 수 있어 정렬 필요
        if os.path.abspath(downloaded) != os.path.abspath(EMBEDDING_FILE):
            os.replace(downloaded, EMBEDDING_FILE)
        print(f"[INFO] 다운로드 완료: {EMBEDDING_FILE}")
        return True
    except Exception as e:
        print(f"[WARN] HF Hub 다운로드 실패: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    기동 흐름:
      1) EMBEDDING_FILE 존재 → 그대로 로드
      2) 없으면 EMBEDDING_HF_REPO 시도 → 받아서 로드
      3) 둘 다 실패 → app.state.game = None, /health 503

    파일이 없어도 서버 자체는 기동된다. 운영자가 컨테이너 로그로 원인 파악 가능,
    파일만 따로 올려도 재시작 한 번이면 복구.
    """
    if not os.path.exists(EMBEDDING_FILE):
        _try_hf_download()

    try:
        print(f"[INFO] Loading embeddings from {EMBEDDING_FILE}...")
        store = load_store(EMBEDDING_FILE)
        app.state.game = GameState(store=store)
        print(f"[INFO] Loaded embeddings ({len(store)} words). Ready.")
    except FileNotFoundError:
        print(f"[ERROR] Embedding file not found: {EMBEDDING_FILE}")
        print("[ERROR] 옵션: data/ 에 직접 파일 배치, 또는 EMBEDDING_HF_REPO 환경변수 설정")
        print("[ERROR] 서버는 기동되지만 게임 라우트는 503 반환")
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
