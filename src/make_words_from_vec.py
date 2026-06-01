import argparse
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")
OUTPUT = os.path.join(DATA_PATH, "words_50000.json")

KOREAN_PATTERN = re.compile(r"^[가-힣]+$")


def is_valid_word(word):
    # 길이 1글자 허용: "끝"/"꿈"/"눈"/"값" 같은 흔한 1글자 명사를
    # 옛 2자 cutoff 가 막아 990개 명사가 누락됐음 (docs/features/05 §1.7 finding 1).
    # 부사 필터 제거: 옛 r".+(게|히)$" 가 너무 광범위해 "가게"/"무게"/"모기" 같은
    # 명사 314개도 같이 잡았음 (finding 2). 진짜 부사("빠르게"·"조용히")는 빈도가
    # cap 50k 안에서 매우 일부라 제거해도 사전 오염 미미.
    if not KOREAN_PATTERN.fullmatch(word):
        return False
    if not (1 <= len(word) <= 6):
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
