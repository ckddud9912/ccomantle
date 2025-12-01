import fasttext
import json
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src 폴더 기준
DATA_PATH = os.path.dirname("../data")

# 🔥 여기만 네 컴퓨터 경로에 맞게 변경
MODEL_PATH = r"C:\Users\창영\Desktop\cc.ko.300.bin"

def is_korean(word):
    # 한글 1자 이상 포함했는지 체크
    return re.search(r"[가-힣]", word) is not None

def main():
    print("🔵 FastText 모델 로드 중...")
    model = fasttext.load_model(MODEL_PATH)
    print("✔ 모델 로드 완료!")

    print("🔵 FastText 단어 리스트 추출 중...")
    words = model.get_words()
    print(f"총 {len(words)} 단어 발견")

    print("🔵 한국어 필터링 중...")
    valid = []
    for w in words:
        if 2 <= len(w) <= 5 and is_korean(w):
            valid.append(w)

    print(f"한국어 후보 단어: {len(valid)}개")

    # 상위 50,000개 자르기
    final = valid[:50000]
    print(f"✔ 최종 50,000개 단어 선택 완료")

    with open(DATA_PATH + "/words_50000.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print("🎉 words_50000.json 생성 완료!")

if __name__ == "__main__":
    main()
