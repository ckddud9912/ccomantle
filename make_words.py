import json

# 5만 단어를 10개 파트로 나누어 내장
# 메시지 크기 제한 때문에 part1부터 part10까지 차례대로 제공됨

parts = []

# ===== 파트 로드 =====
# ChatGPT가 순서대로 제공할 part1~part10 내용을
# 아래 리스트에 append 하는 구조로 생성됩니다.

# part1 ~ part10을 아래에서 연속해서 붙여넣게 됩니다.
# 지금은 빈 리스트로 두고 이후 메시지에서 제공될 파트 텍스트를
# 각각 append 시켜서 구성하게 됩니다.

# 예시:
# parts.append(["가게","가격",...])  # part1
# parts.append([...])               # part2
# ...
# parts.append([...])               # part10


# ===== 파일 생성 =====
words = []
for p in parts:
    words.extend(p)

print("[make_words] 총 단어 수:", len(words))

with open("words_50000.json", "w", encoding="utf-8") as f:
    json.dump(words, f, ensure_ascii=False, indent=2)

print("[make_words] words_50000.json 생성 완료!")
