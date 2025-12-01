import os
import subprocess

ASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src 폴더 기준
DATA_PATH = os.path.dirname("../data")


# ============================
# 1) words_50000.json 먼저 생성
# ============================
if not os.path.exists(DATA_PATH + "/words_50000.json"):
    print("[자동 실행 - run_embed] words_50000.json 없음 → make_words.py 실행")
    subprocess.run(["python", "make_words.py"], check=True)
    print("[자동 실행 - run_embed] make_words.py 완료")


# ============================
# 2) 임베딩 사전 생성 실행
# ============================
if not os.path.exists(DATA_PATH + "/embedding_dictionary.json"):
    print(
        "[INFO] embedding_dictionary.json not found. Running embedding_precompute.py..."
    )
    subprocess.run(["python", "embedding_precompute.py"], check=True)
    print("[INFO] run_embed.py finished.")
else:
    print("[INFO] embedding_dictionary.json already exists.")
