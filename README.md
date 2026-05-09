---
title: Ccomantle
emoji: 🌖
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 시그널 (Ccomantle)

한국어 단어 의미 유사도 추측 게임. 운영자가 정답 단어를 정하고, 팀들이 라운드별로
추측 단어를 제출해 코사인 유사도를 경쟁합니다. QR 코드로 모바일에서 바로 참가 가능.

## 빠른 시작 (운영자)

### 1. 임베딩 파일 준비
서버는 약 5만 개 한국어 단어의 사전계산된 임베딩 파일이 필요합니다.

```
data/embedding_dictionary_e5.json
```

위 경로에 배치하거나 환경변수로 다른 경로를 지정합니다.

```bash
export EMBEDDING_FILE=/path/to/embedding_dictionary_e5.json
```

> 임베딩 파일 직접 생성은 [docs/features/04_preprocessing.md](docs/features/04_preprocessing.md) 참고.
> 파일이 없으면 서버는 기동되지만 게임 라우트가 503을 반환합니다 (`/health`로 확인).

### 2. 로컬 실행

```bash
pip install -r requirements.txt
python src/app.py
```

| 페이지 | URL |
|---|---|
| 메인 | http://localhost:7860/ |
| 게임 | http://localhost:7860/game |
| 관리자 | http://localhost:7860/admin |
| 헬스체크 | http://localhost:7860/health |

### 3. QR 공유 흐름
1. 관리자 페이지(`/admin`)에서 정답 단어 설정
2. 게임 페이지(`/game`) URL을 QR 코드로 변환
3. 참가자가 모바일로 QR 스캔 → 팀명/색상 입력 → 단어 제출

## 문서
- [docs/](docs/) — 기능별 상세 문서, 변경 이력
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — 리팩토링 작업 이력
- [docs/features/03_api.md](docs/features/03_api.md) — API 레퍼런스

## 라이선스
MIT
