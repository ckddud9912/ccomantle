---
title: Ccomantle
emoji: 🌖
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# 시그널 (Ccomantle) 🌖

> **한국어 단어 의미 유사도 추측 게임 — 팀전 버전.**
> 운영자가 정답 단어 한 개를 정하고, 팀들이 6라운드 안에 의미 가까운 단어를 제출해 경쟁합니다. 의미가 가까울수록 코사인 유사도가 높고, 라운드별 점수와 최종 평균으로 순위가 결정됩니다.
> **QR 코드로 모바일에서 바로 참가** — 워크샵·교육·아이스브레이킹 즉시 사용 가능.

---

## ✨ 주요 기능

- 🎯 **6라운드 팀전 게임** — 정답에 가까울수록 높은 점수, 6라운드 평균으로 최종 순위
- 📱 **QR 모바일 즉시 참가** — 별도 앱 설치 X, 브라우저만으로
- 👥 **실시간 리더보드** — 팀명·팀색 입력, 내 팀 행 하이라이트, 막대 색상 그라디언트 (가까울수록 초록)
- 🛠 **운영자 페이지** — 정답 설정 · 라운드 제어 · 경기 종료 · 게임 재시작 · 참여 팀 누락 검사 · top1000 단어 조회
- 🔬 **한국어 60,000 단어 사전계산 임베딩** ([KoE5](https://huggingface.co/nlpai-lab/KoE5) — 한국어 특화 모델, passage prefix) + 외부 사전 (mecab/우리말샘) 으로 어휘 보강. 저장 형식 NPZ binary (~235MB)
- 🐳 **`docker compose up` 한 줄 실행** + `.env` 자동 로드 (도커/비도커 양쪽)

---

## 🚀 빠른 시작

> **📦 임베딩 파일이 필요합니다** (한국어 60,000 단어 사전계산 벡터, NPZ ~235MB).
> 아래 두 가지 경로 중 하나로 받으세요. 둘 다 안 한 상태로 실행하면 서버는 뜨지만 게임 라우트가 503을 반환합니다 (`/health`로 확인 가능).

### 옵션 A: 도커로 실행 (권장)

#### 1. 임베딩 파일 준비 — A-1 또는 A-2 중 택일

**A-1. HuggingFace Hub에서 자동 다운로드 (권장)**

`.env` 파일을 만들고 임베딩이 올라간 HF dataset 경로를 적으세요.

```bash
cp .env.example .env
# .env 파일을 열어 EMBEDDING_HF_REPO 값 입력. 예:
# EMBEDDING_HF_REPO=leon4study/ccomantle-embeddings
```

컨테이너 첫 기동 시 자동 다운로드 후 `data/` 에 캐싱됩니다.

**A-2. 파일을 직접 받아 두기**

`data/embedding_dictionary_e5.npz` 위치에 파일을 직접 두세요 (옛 `.json` 도 지원하지만 `.npz` 가 약 80% 작음). 도커가 이 폴더를 컨테이너에 마운트하므로 `.env` 설정은 필요 없습니다.

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

# .env 파일은 자동 로드됨 (옵션 A와 동일하게 cp .env.example .env)
python src/app.py
```

http://localhost:7860 으로 접속.

> 도커/비도커 어느 쪽이든 `.env` 파일이 있으면 자동으로 환경변수로 읽힙니다. 이미 export 된 환경변수가 있으면 그게 우선됨 (.env가 덮어쓰지 않음).

---

## 🎮 운영 흐름

```
1. 운영자가 /admin 에서 정답 단어 설정
2. /game URL 을 QR 로 변환 (qr-code-generator.com 등)
3. 참가자가 모바일로 QR 스캔
4. 팀명 / 색상 입력 후 라운드별 단어 제출
5. 운영자가 라운드 진행 (참여 팀 누락 시 confirm 으로 보호)
6. 6라운드 종료 또는 운영자가 "경기 종료" 클릭
7. 정답 공개 + 단어 랭킹 + 팀별 평균 유사도 표시
8. (선택) "게임 재시작" 으로 같은 팀들 그대로 새 정답으로 한 판 더
```

### 활용 시나리오 예시
- 사내 워크샵·교육 아이스브레이커 (10-50명, 30분)
- 한국어 어휘·의미 학습 도구
- 팀 빌딩 미니 게임
- LLM·임베딩 동작 데모

---

## 🧱 기술 스택

| 영역 | 도구 |
|---|---|
| Backend | FastAPI + asyncio (단일 워커, asyncio.Lock 으로 동시 제출 직렬화) |
| Frontend | 정적 HTML/CSS/JS (vanilla, 모바일 우선) |
| 임베딩 | [KoE5](https://huggingface.co/nlpai-lab/KoE5) (multilingual-e5 한국어 finetune) 60,000 단어 사전계산 (L2 normalized, float32, NPZ binary). FastText 50k 기반 + mecab-ko-dic·우리말샘 보강 누락 명사 ~10,000 + 한국어 의미 cluster 정상화 (PR #21·#22) |
| 인프라 | Docker Compose + `.env` 자동 로드 + HF Hub 자동 다운로드 + `/health` graceful startup |

자세한 구조는 [docs/features/](docs/features/) 참고.

---

## ⚙️ 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `EMBEDDING_HF_REPO` | (없음) | HF Hub repo. 비어있으면 다운로드 시도 안 함 |
| `EMBEDDING_HF_FILE` | `embedding_dictionary_e5.npz` | HF 내 파일명 (옛 `.json` 도 지원, `.npz` 가 약 80% 작음) |
| `EMBEDDING_HF_TYPE` | `dataset` | `dataset` / `model` / `space` |
| `EMBEDDING_FILE` | 자동 (`.npz` 우선 → `.json` fallback) | 로컬 임베딩 파일 경로. 명시 안 하면 `data/embedding_dictionary_e5.npz` 우선 시도, 없으면 `.json` |

---

## 🛠 임베딩 파일 직접 생성하기

기존 파일을 받지 않고 처음부터 만들고 싶다면 두 path 중 선택:

### Path A: FastText 부터 (전통, 시간 듦)

```bash
pip install -r requirements-dev.txt

# 1. FastText 한국어 .vec 파일 다운로드 (1.6GB)
#    https://fasttext.cc/docs/en/crawl-vectors.html

# 2. 단어 추출 (FastText 상위 50,000)
export FASTTEXT_VEC_PATH=/실제/받은/경로/cc.ko.300.vec   # ← 실제 경로로 교체
python src/make_words_from_vec.py

# 3. 임베딩 생성 — KoE5 (한국어 특화, passage prefix)
python tools/embedding_eval/build_alt_embeddings.py \
  --model nlpai-lab/KoE5 \
  --output data/embedding_dictionary_e5.json \
  --prefix "passage: "

# 4. (선택) 사전 어휘 보강 — mecab-ko-dic / 우리말샘 사전과 diff
python tools/dict_quality/coverage_check.py    # mecab vs 본인 사전
python tools/dict_quality/diff_urimalsaem.py   # 우리말샘 vs 본인 사전 (POS 별)
python tools/dict_quality/expand_dict.py       # 후보 단어 임베딩 추가 머지

# 5. (권장) JSON → NPZ 변환 (~80% 절약)
python tools/storage/convert_to_npz.py
```

### Path B: 기존 사전 받아서 + 보강만

HF Hub 의 60k 사전 (`leo4study/ccomantle-embeddings`) 받아두면 위 1-3 단계 skip 가능. 이후 4-5 만 진행.

### 임베딩 모델 비교 / 교체

대안 모델 비교 (KoE5 vs KURE 등):
```bash
python tools/embedding_eval/build_alt_embeddings.py --model nlpai-lab/KURE-v1 --prefix "" --output data/embedding_dictionary_kure.json
# notebooks/embedding_model_comparison.ipynb 또는 embedding_tuning_comparison.ipynb 에서 spot-check
# apply_scaling_and_swap.py 로 swap (scaling 시도 후 raw 그대로 가능)
```

자세한 내용은 [docs/features/04_preprocessing.md](docs/features/04_preprocessing.md), 평가 방법론은 [docs/features/05_evaluation_methodology.md](docs/features/05_evaluation_methodology.md) 참고.

> 🟢 **사전 품질 진행 상황 (PR #15-#23 누적)**:
> - 어휘: ko-FastText 50,000 단어 → mecab+우리말샘 누락 명사 보강으로 **60,000 단어**
> - 모델: multilingual-e5-large → **KoE5 (한국어 특화) passage prefix** — 과일 cross-check 정상화 (PR #21)·prefix 변경 (PR #22)
> - 형식: JSON 1.3GB → **NPZ 235MB (82% 절약)** (PR #23)
> - 1글자 명사 ("끝"·"꿈") + 부사 패턴 명사 ("가게"·"무게") 다수 회복. 게임에서 "사과" 정답 시 과일 4개 (수박·딸기·포도·배) top 4 차지
> - 다음 트랙: 활용형 중복 정리, sim_alpha 보정 강화, KorLex Golden 정량 평가

---

## 🩹 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `/health` 가 503 반환 | 임베딩 파일 미로드. 컨테이너 로그 확인 → 파일 위치 / `EMBEDDING_HF_REPO` 점검 |
| `docker compose up` 빌드 실패 | Python 3.10+ 도커 이미지 받기 위해 인터넷 연결 확인 |
| 정적 파일 변경이 도커에 안 보임 | `docker compose up --build` 로 이미지 재빌드 필요 (`static/` 은 볼륨 마운트 안 됨) |
| 모바일에서 입력칸 폰트 작음 | iOS Safari 자동 줌 방지 위해 16px+ 적용됨. 캐시 삭제 후 재접속 |
| 한글 입력 후 Enter 작동 안 함 | IME 조합 완료(스페이스/조합 종료) 후 Enter 누르면 됨 |
| 같은 팀이 여러 번 제출됨 | `/guess` 응답이 `duplicate` 인지 확인. 라운드 변경 시 다시 제출 가능 |
| 새로고침 후 팀 색상이 파랑으로 리셋 | 강제 새로고침(Cmd/Ctrl+Shift+R) 으로 캐시된 옛 JS 갱신 |

---

## 📚 문서

- [docs/](docs/) — 기능별 상세 문서, 변경 이력
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — 작업 이력 (최신이 위)
- [docs/features/01_embeddings.md](docs/features/01_embeddings.md) — 임베딩 스토어 구조
- [docs/features/02_game.md](docs/features/02_game.md) — 게임 로직·동시성
- [docs/features/03_api.md](docs/features/03_api.md) — API 레퍼런스
- [docs/features/04_preprocessing.md](docs/features/04_preprocessing.md) — 임베딩 생성 파이프라인

---

## 라이선스

MIT
