import argparse
import json
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")
OUTPUT_FILE = os.path.join(DATA_PATH, "words_50000.json")


def _resolve(cli_val, env_key, label):
    path = cli_val or os.getenv(env_key, "")
    if not path:
        sys.exit(f"[ERROR] {label} 경로 필요: 해당 옵션 또는 환경변수 {env_key}")
    return path


def parse_args():
    p = argparse.ArgumentParser(description="fasttext dump vocab → words_50000.json")
    p.add_argument("--exe", default=None, help="fasttext 실행파일 경로 (또는 FASTTEXT_EXE_PATH)")
    p.add_argument("--model", default=None, help="FastText .bin 경로 (또는 FASTTEXT_MODEL_PATH)")
    p.add_argument("--output", default=OUTPUT_FILE)
    return p.parse_args()


def extract_words(fasttext_exe, model_path, output_file):
    if not os.path.exists(fasttext_exe):
        sys.exit(f"[ERROR] fasttext 실행파일 없음: {fasttext_exe}")
    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] FastText 모델 없음: {model_path}")

    cmd = [fasttext_exe, "dump", "vocab", model_path]
    print("실행:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"[ERROR] FastText 실행 오류:\n{result.stderr}")

    words = [line.strip().split()[0] for line in result.stdout.split("\n") if line.strip()]
    print(f"전체 단어 수: {len(words)}")

    top_50k = words[:50000]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(top_50k, f, ensure_ascii=False, indent=2)

    print(f"저장 완료 → {output_file} ({len(top_50k)}개)")


def main():
    args = parse_args()
    fasttext_exe = _resolve(args.exe, "FASTTEXT_EXE_PATH", "fasttext 실행파일")
    model_path = _resolve(args.model, "FASTTEXT_MODEL_PATH", "FastText .bin 모델")
    extract_words(fasttext_exe, model_path, args.output)


if __name__ == "__main__":
    main()
