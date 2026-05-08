import argparse
import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")


def is_korean(word):
    return re.search(r"[가-힣]", word) is not None


def _resolve(cli_val, env_key, label):
    path = cli_val or os.getenv(env_key, "")
    if not path:
        sys.exit(f"[ERROR] {label} 경로 필요: --model 또는 환경변수 {env_key}")
    return path


def parse_args():
    p = argparse.ArgumentParser(description="FastText .bin → words_50000.json")
    p.add_argument("--model", default=None, help="FastText .bin 경로 (또는 FASTTEXT_MODEL_PATH)")
    p.add_argument("--output", default=os.path.join(DATA_PATH, "words_50000.json"))
    return p.parse_args()


def main():
    args = parse_args()
    model_path = _resolve(args.model, "FASTTEXT_MODEL_PATH", "FastText .bin 모델")

    import fasttext
    print("FastText 모델 로드 중...")
    model = fasttext.load_model(model_path)
    print("모델 로드 완료!")

    words = model.get_words()
    print(f"총 {len(words)} 단어 발견")

    valid = [w for w in words if 2 <= len(w) <= 5 and is_korean(w)]
    print(f"한국어 후보 단어: {len(valid)}개")

    final = valid[:50000]
    print(f"최종 {len(final)}개 단어 선택 완료")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"저장 완료 → {args.output}")


if __name__ == "__main__":
    main()
