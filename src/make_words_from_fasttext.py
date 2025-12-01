import subprocess
import json
import os

# ▶ FastText 실행 파일 (네 PC 경로)
FASTTEXT_EXE = r"C:\Users\창영\Desktop\fasttext\fasttext.exe"

# ▶ 한국어 FastText 모델 경로 (.bin 파일)
MODEL_PATH = r"C:\Users\창영\Desktop\fasttext\cc.ko.300.bin"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src 폴더 기준
DATA_PATH = os.path.dirname("../data")

# ▶ 출력 파일
OUTPUT_FILE = DATA_PATH + "/words_50000.json"


def extract_words():
    print("🔵 FastText에서 단어 목록 추출 중...")

    # -------------------------------
    # 경로 유효성 체크
    # -------------------------------
    if not os.path.exists(FASTTEXT_EXE):
        print(f"❌ fasttext.exe 파일을 찾을 수 없음:\n  {FASTTEXT_EXE}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ FastText 모델 파일 없음:\n  {MODEL_PATH}")
        return

    # -------------------------------
    # FastText 단어 목록 추출
    # -------------------------------
    cmd = [
        FASTTEXT_EXE,
        "dump",
        "vocab",
        MODEL_PATH
    ]

    print("📌 실행:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ FastText 실행 오류:")
        print(result.stderr)
        return

    lines = result.stdout.split("\n")

    words = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            words.append(parts[0])

    print(f"📌 전체 단어 수: {len(words)}")

    # 상위 50,000개만 사용
    top_50k = words[:50000]

    # -------------------------------
    # 저장
    # -------------------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(top_50k, f, ensure_ascii=False, indent=2)

    print(f"🎉 50,000 단어 저장 완료 → {OUTPUT_FILE}")
    print(f"총 {len(top_50k)} 개 단어가 저장되었습니다.")


if __name__ == "__main__":
    extract_words()
