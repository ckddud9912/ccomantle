import argparse
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
MAKE_WORDS_SCRIPT = os.path.join(BASE_DIR, "make_words_from_vec.py")
EMBED_SCRIPT = os.path.join(BASE_DIR, "E5_embedding_ver2.py")


def parse_args():
    p = argparse.ArgumentParser(description="단어 추출 → 임베딩 생성 파이프라인")
    p.add_argument("--vec", default=None, help=".vec 파일 경로 (또는 FASTTEXT_VEC_PATH)")
    p.add_argument("--force", action="store_true", help="기존 파일 있어도 재실행")
    return p.parse_args()


def run(cmd, **kwargs):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    args = parse_args()

    vec_path = args.vec or os.getenv("FASTTEXT_VEC_PATH", "")

    # 1) words_50000.json
    words_file = os.path.join(DATA_DIR, "words_50000.json")
    if args.force or not os.path.exists(words_file):
        if not vec_path:
            sys.exit("[ERROR] words_50000.json 없음. --vec 또는 FASTTEXT_VEC_PATH 환경변수 필요")
        run(
            [sys.executable, MAKE_WORDS_SCRIPT, "--vec", vec_path],
            env={**os.environ, "FASTTEXT_VEC_PATH": vec_path},
        )
    else:
        print(f"[SKIP] {words_file} 이미 존재")

    # 2) embedding_dictionary_e5_scaled.json
    embed_file = os.path.join(DATA_DIR, "embedding_dictionary_e5_scaled.json")
    if args.force or not os.path.exists(embed_file):
        run([sys.executable, EMBED_SCRIPT])
    else:
        print(f"[SKIP] {embed_file} 이미 존재")

    print("\n[완료] 파이프라인 종료")


if __name__ == "__main__":
    main()
