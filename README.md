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
추측 단어를 제출해 코사인 유사도를 경쟁합니다. **QR 코드로 모바일에서 바로 참가 가능**.

---

## 빠른 시작

> **📦 임베딩 파일이 필요합니다** (한국어 5만개 사전계산 벡터, 수백 MB).
> 아래 두 가지 경로 중 하나로 받으세요. 둘 다 안 한 상태로 실행하면 서버는
> 뜨지만 게임 라우트가 503을 반환합니다 (`/health`로 확인 가능).

### 옵션 A: 도커로 실행 (권장)

#### 1. 임베딩 파일 준비 — A-1 또는 A-2 중 택일

**A-1. HuggingFace Hub에서 자동 다운로드 (권장)**

`.env` 파일을 만들고 본인이 임베딩을 올려둔 HF dataset 경로를 적으세요.

```bash
cp .env.example .env
# .env 파일을 열어 EMBEDDING_HF_REPO 값 입력. 예:
# EMBEDDING_HF_REPO=leon4study/ccomantle-embeddings
```

컨테이너 첫 기동 시 자동 다운로드 후 `data/` 에 캐싱됩니다.

**A-2. 파일을 직접 받아 두기**

`data/embedding_dictionary_e5.json` 위치에 파일을 직접 두세요. 도커가 이 폴더를
컨테이너에 마운트하므로 `.env` 설정은 필요 없습니다.

#### 2. 실행

```bash
docker compose up
```

| 페이지 | URL |
|---|---|
| 메인 | http://localhost:7860/ |
| 게임 | http://localhost:7860/game |
| 관리자 | http://localhost:7860/admin |
| 헬스체크 | http://localhost:7860/health |

종료: `Ctrl+C` 또는 `docker compose down`

---

### 옵션 B: 도커 없이 직접 실행

```bash
# Python 3.10+ 권장
pip install -r requirements.txt

# 임베딩 파일 처리는 옵션 A와 동일 (data/ 에 두거나 환경변수 설정)
export EMBEDDING_HF_REPO=leon4study/ccomantle-embeddings   # 선택

python src/app.py
```

http://localhost:7860 으로 접속.

---

## 운영 흐름

```
1. 운영자가 /admin 에서 정답 단어 설정
2. /game URL 을 QR 로 변환 (qr-code-generator.com 등)
3. 참가자가 모바일로 QR 스캔
4. 팀명 / 색상 입력 후 라운드별 단어 제출
5. 6라운드 종료 후 운영자가 "경기 종료" 클릭
6. 정답 공개 + 팀별 평균 유사도 순위 표시
```

---

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `EMBEDDING_HF_REPO` | (없음) | HF Hub repo. 비어있으면 다운로드 시도 안 함 |
| `EMBEDDING_HF_FILE` | `embedding_dictionary_e5.json` | HF 내 파일명 |
| `EMBEDDING_HF_TYPE` | `dataset` | `dataset` / `model` / `space` |
| `EMBEDDING_FILE` | `data/embedding_dictionary_e5.json` | 로컬 임베딩 파일 절대경로 |

---

## 임베딩 파일 직접 생성하기

기존 파일을 받지 않고 처음부터 만들고 싶다면:

```bash
pip install -r requirements-dev.txt

# 1. FastText 한국어 .vec 파일 다운로드 (1.6GB)
#    https://fasttext.cc/docs/en/crawl-vectors.html

# 2. 단어 추출 (5만 개)
export FASTTEXT_VEC_PATH=/path/to/cc.ko.300.vec
python src/make_words_from_vec.py

# 3. E5 임베딩 생성 (GPU 권장, CPU 시 30분~)
python src/E5_embedding_ver2.py
```

자세한 내용은 [docs/features/04_preprocessing.md](docs/features/04_preprocessing.md) 참고.

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `/health` 가 503 반환 | 임베딩 파일 미로드. 컨테이너 로그 확인 → 파일 위치 / `EMBEDDING_HF_REPO` 점검 |
| `docker compose up` 빌드 실패 | Python 3.10+ 도커 이미지 받기 위해 인터넷 연결 확인 |
| 모바일에서 입력칸 폰트 작음 | iOS Safari 자동 줌 방지 위해 16px+ 적용됨. 캐시 삭제 후 재접속 |
| 한글 입력 후 Enter 작동 안 함 | IME 조합 완료(스페이스/조합 종료) 후 Enter 누르면 됨 |
| 같은 팀이 여러 번 제출됨 | `/guess` 응답이 `duplicate` 인지 확인. 라운드 변경 시 다시 제출 가능 |

---

## 문서
- [docs/](docs/) — 기능별 상세 문서, 변경 이력
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — 리팩토링 작업 이력
- [docs/features/03_api.md](docs/features/03_api.md) — API 레퍼런스
- [docs/features/04_preprocessing.md](docs/features/04_preprocessing.md) — 임베딩 생성 파이프라인

## 라이선스
MIT
