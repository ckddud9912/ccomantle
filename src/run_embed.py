import os
import subprocess

# 이 파일(run_embed.py)이 위치한 src 폴더의 절대 경로를 정의합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# data 폴더의 절대 경로를 올바르게 계산합니다. (src 폴더에서 상위 폴더의 data)
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
# make_words_from_vec.py 및 embedding_precompute.py의 절대 경로를 만듭니다.
MAKE_WORDS_PATH = os.path.join(BASE_DIR, "make_words_from_vec.py")
EMBED_PRECOMPUTE_PATH = os.path.join(BASE_DIR, "embedding_precompute.py")

# 🌟 진단용 출력: 경로 확인 (실행 흐름 파악)
print(f"[run_embed DIAG] BASE_DIR: {BASE_DIR}")
print(f"[run_embed DIAG] DATA_DIR: {DATA_DIR}")


# ============================
# 1) words_50000.json 먼저 생성
# ============================
WORDS_FILE_PATH = os.path.join(DATA_DIR, "words_50000.json")
print(f"[run_embed DIAG] Checking for words file at: {WORDS_FILE_PATH}")

if not os.path.exists(WORDS_FILE_PATH):
    # 이 구문은 파일이 실제로 없을 때만 실행되어야 합니다.
    print(WORDS_FILE_PATH)
    print("[자동 실행 - run_embed] words_50000.json 없음 → make_words_from_vec.py 실행")
    # words_50000.json이 없다면 make_words_from_vec.py가 실행됩니다.
    subprocess.run(["python", MAKE_WORDS_PATH], check=True)
    print("[자동 실행 - run_embed] make_words_from_vec.py 완료")
else:
    # 🌟 파일이 존재하므로 이 구문이 출력되어야 합니다.
    print("[INFO] words_50000.json already exists. Skipping word extraction.")


# ============================
# 2) 임베딩 사전 생성 실행
# ============================
EMBEDDING_FILE_PATH = os.path.join(DATA_DIR, "embedding_dictionary.json")
print(f"[run_embed DIAG] Checking for embedding file at: {EMBEDDING_FILE_PATH}")

if not os.path.exists(EMBEDDING_FILE_PATH):
    print(
        "[INFO] embedding_dictionary.json not found. Running embedding_precompute.py..."
    )
    subprocess.run(
        ["python", EMBED_PRECOMPUTE_PATH],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print("[INFO] run_embed.py finished.")
else:
    print(
        "[INFO] embedding_dictionary.json already exists. Skipping embedding generation."
    )
