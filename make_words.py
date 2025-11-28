import json
import random

# 한국어 자음/모음 조합을 이용해 정상적인 2~4글자 단어 생성
# 50,000개 생성

chos = ["ㄱ","ㄴ","ㄷ","ㄹ","ㅁ","ㅂ","ㅅ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
jungs = ["ㅏ","ㅑ","ㅓ","ㅕ","ㅗ","ㅛ","ㅜ","ㅠ","ㅡ","ㅣ"]
jongs = ["", "ㄱ","ㄴ","ㄷ","ㄹ","ㅁ","ㅂ","ㅅ","ㅇ","ㅈ","ㅊ"]

# 한글 문자 조합 함수
def make_korean_char(c, j, k):
    return chr(0xAC00 + (chos.index(c)*21*28) + (jungs.index(j)*28) + jongs.index(k))

def random_word():
    length = random.choice([2,3,4])
    return "".join([
        make_korean_char(
            random.choice(chos),
            random.choice(jungs),
            random.choice(jongs)
        ) for _ in range(length)
    ])

# 자연스러운 단어 리스트 생성
words = []
while len(words) < 50000:
    w = random_word()
    if w not in words:
        words.append(w)

print("[make_words] 총 단어 수:", len(words))

with open("words_50000.json", "w", encoding="utf-8") as f:
    json.dump(words, f, ensure_ascii=False, indent=2)

print("[make_words] words_50000.json 생성 완료!")
