import json

# 🔥 .vec 파일 경로 (압축 풀린 파일)
VEC_PATH = r"C:\Users\창영\Desktop\fasttext\cc.ko.300.vec"

# 🔥 결과 저장 파일
OUTPUT = "words_50000.json"


def extract_words():
    print("🔵 .vec 파일에서 단어 추출 중...")

    words = []

    with open(VEC_PATH, "r", encoding="utf-8", errors="ignore") as f:
        # 첫 줄은 "단어수 차원수" 메타데이터 → 버린다
        header = f.readline()
        print("📌 헤더:", header.strip())

        # 나머지 줄을 읽으며 단어 추출
        for i, line in enumerate(f):
            if i >= 50000:   # 상위 50,000개만
                break
            
            parts = line.split()
            if not parts:
                continue
            
            word = parts[0]
            words.append(word)

    print(f"📌 추출된 단어 수: {len(words)}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    print(f"🎉 저장 완료 → {OUTPUT}")


if __name__ == "__main__":
    extract_words()
