import json
import re

# 🔥 .vec 파일 경로 (압축 풀린 파일)
VEC_PATH = r"C:\Users\창영\Desktop\fasttext\cc.ko.300.vec"

# 🔥 결과 저장 파일
OUTPUT = "words_50000.json"

# ---------------------------
# 🔍 정규식 필터 정의
# ---------------------------

# 한국어만 허용 (초성·중성 단독 불가)
KOREAN_PATTERN = re.compile(r"^[가-힣]+$")

# 부사 패턴 (게, 히로 끝나는 단어 제외)
ADVERB_PATTERN = re.compile(r".+(게|히)$")


def is_valid_word(word):
    """모든 필터 조건을 만족하는지 검사"""

    # 한국어만 포함
    if not KOREAN_PATTERN.fullmatch(word):
        return False

    # 글자 수 제한
    if not (2 <= len(word) <= 6):
        return False

    # 부사 제외 (게/히로 끝나는 단어)
    if ADVERB_PATTERN.fullmatch(word):
        return False

    return True


def extract_words():
    print("🔵 .vec 파일에서 단어 추출 중...\n")

    filtered_words = []
    seen = set()

    with open(VEC_PATH, "r", encoding="utf-8", errors="ignore") as f:
        # 첫 줄 (메타 정보) 제거
        header = f.readline().strip()
        print(f"📌 헤더: {header}")

        for line_num, line in enumerate(f):
            if len(filtered_words) >= 50000:
                break

            parts = line.split()
            if not parts:
                continue

            word = parts[0]

            # 중복 제거
            if word in seen:
                continue

            # 필터 검사
            if not is_valid_word(word):
                continue

            # 단어 추가
            filtered_words.append(word)
            seen.add(word)

            # 진행 상황 출력 (1000 단위)
            if len(filtered_words) % 1000 == 0:
                print(f"   → {len(filtered_words)}개 수집 중...")

    print("\n📌 최종 수집된 단어:", len(filtered_words))

    # JSON 저장
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(filtered_words, f, ensure_ascii=False, indent=2)

    print(f"🎉 저장 완료 → {OUTPUT}")


if __name__ == "__main__":
    extract_words()
