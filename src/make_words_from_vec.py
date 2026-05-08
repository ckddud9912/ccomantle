import argparse
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")
OUTPUT = os.path.join(DATA_PATH, "words_50000.json")

KOREAN_PATTERN = re.compile(r"^[가-힣]+$")
ADVERB_PATTERN = re.compile(r".+(게|히)$")


def is_valid_word(word):
    if not KOREAN_PATTERN.fullmatch(word):
        return False
    if not (2 <= len(word) <= 6):
        return False
    if ADVERB_PATTERN.fullmatch(word):
        return False
    return True


def _resolve(cli_val, env_key, label):
    path = cli_val or os.getenv(env_key, "")
    if not path:
        sys.exit(f"[ERROR] {label} 경로 필요: --vec 또는 환경변수 {env_key}")
    return path


def parse_args():
    p = argparse.ArgumentParser(description=".vec 파일 → words_50000.json")
    p.add_argument("--vec", default=None, help=".vec 파일 경로 (또는 FASTTEXT_VEC_PATH)")
    p.add_argument("--output", default=OUTPUT)
    return p.parse_args()


def extract_words(vec_path, output_file):
    filtered_words = []
    seen = set()

    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip()
        print(f"헤더: {header}")

        for line in f:
            if len(filtered_words) >= 50000:
                break

            parts = line.split()
            if not parts:
                continue

            word = parts[0]
            if word in seen or not is_valid_word(word):
                continue

            filtered_words.append(word)
            seen.add(word)

            if len(filtered_words) % 5000 == 0:
                print(f"   → {len(filtered_words)}개 수집 중...")

    print(f"최종 수집된 단어: {len(filtered_words)}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_words, f, ensure_ascii=False, indent=2)

    print(f"저장 완료 → {output_file}")


def main():
    args = parse_args()
    vec_path = _resolve(args.vec, "FASTTEXT_VEC_PATH", ".vec 파일")
    extract_words(vec_path, args.output)


if __name__ == "__main__":
    main()
