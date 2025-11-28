import urllib.request
import json

URL = "https://raw.githubusercontent.com/kkomentle-data/ko-words/main/words_50000.json"

print("[다운로드] 50,000 단어 사전 다운로드 시작...")
try:
    data = urllib.request.urlopen(URL).read().decode("utf-8")
    words = json.loads(data)

    with open("words_50000.json", "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    print("[완료] words_50000.json 생성 완료! 단어 수:", len(words))

except Exception as e:
    print("[오류] 다운로드 실패:", e)
