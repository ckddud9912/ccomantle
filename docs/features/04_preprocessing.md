# 전처리 / 사전·임베딩 사이클

> 게임 서버가 사용하는 `data/embedding_dictionary_e5.{npz,json}` 을 만들고 개선하는 도구·흐름. 1회성 추출이 아니라 **진단·보강·교체·최적화 사이클**.

## 전체 그림

```
┌─────────────────┐
│  source 단어     │  FastText / mecab-ko-dic / 우리말샘 / ko 위키
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  사전 어휘       │  data/words_NNNNN.json   ← src/make_words_from_vec.py
│  (60,000 단어)   │                            tools/dict_quality/expand_dict.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  임베딩 벡터     │  data/embedding_dictionary_e5.{npz,json}
│  (60k × 1024 D) │   ← src/E5_embedding_ver2.py
│                  │     tools/embedding_eval/build_alt_embeddings.py
│  KoE5 passage    │     tools/embedding_eval/apply_scaling_and_swap.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  저장 형식       │  NPZ (~235MB) 또는 JSON (~1.3GB)
│                  │   ← tools/storage/convert_to_npz.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HF Hub 배포     │  leo4study/ccomantle-embeddings
└─────────────────┘
```

각 단계에서 진단·교체·튜닝 사이클이 따로 돔. PR #15-#23 누적.

---

## 1. 단어 추출 (사전 어휘)

### 1.1 FastText 기반 (전통)

[`src/make_words_from_vec.py`](../../src/make_words_from_vec.py) — ko-FastText `.vec` 파일의 상위 50,000 토큰 추출. 필터:
- 순수 한국어 (`[가-힣]+$`)
- 길이 1-6 (PR #16 이후 1글자 허용)
- 부사 필터 제거 (PR #16 이전엔 "-게/-히" 끝 단어 거부했으나 명사도 잡혀서 제거)

```bash
export FASTTEXT_VEC_PATH=/실제/경로/cc.ko.300.vec
python src/make_words_from_vec.py
# → data/words_50000.json
```

### 1.2 외부 사전 비교 + 누락 보강 (도구 사이클)

FastText 50k 가 한국어 명사 커버리지 약 5-7% 라 외부 사전과 비교 후 누락 단어 추가:

| 도구 | reference | 산출 |
|---|---|---|
| [`tools/dict_quality/coverage_check.py`](../../tools/dict_quality/coverage_check.py) | mecab-ko-dic NNG (~200k) | coverage % + missing 후보 |
| [`tools/dict_quality/diff_urimalsaem.py`](../../tools/dict_quality/diff_urimalsaem.py) | 국립국어원 우리말샘 (~1.2M, POS 별) | POS 별 missing JSON + MD 리포트 |
| [`tools/dict_quality/diff_kowiki.py`](../../tools/dict_quality/diff_kowiki.py) | ko 위키 article titles | (참고용, 일본 지명·인명 다수라 노이즈) |
| [`tools/dict_quality/expand_dict.py`](../../tools/dict_quality/expand_dict.py) | 후보 JSON 받아 새 단어 임베딩 추가 머지 | `data/words_NNNNN.json` 확장 |

```bash
# 진단
python tools/dict_quality/coverage_check.py
python tools/dict_quality/diff_urimalsaem.py

# 보강 — mecab+우리말샘 교집합 명사 top 5,000 추가
python tools/dict_quality/expand_dict.py \
  --source data/missing_words_urimalsaem_nouns.json \
  --filter-source "urimalsaem+mecab" \
  --top-n 5000
```

PR #17 가 mecab 5k 보강 (50k → 55k), PR #19 가 우리말샘 5k 보강 (55k → 60k).

### 1.3 알려진 한계

- mecab cost 가 균질 (top 5,000 = cost 2,628-2,639) → "자두"·"귤"·"참외" 같이 매우 흔한 단어가 156,000+위로 떨어져 top N 으론 못 잡힘
- 다음 트랙: ko 위키 article body 빈도 source (실제 사용 빈도)

---

## 2. 임베딩 생성

### 2.1 현행 — KoE5 (한국어 특화)

[`tools/embedding_eval/build_alt_embeddings.py`](../../tools/embedding_eval/build_alt_embeddings.py) — 임의 HF 모델로 임베딩 생성. Apple Silicon MPS 자동.

```bash
python tools/embedding_eval/build_alt_embeddings.py \
  --model nlpai-lab/KoE5 \
  --output data/embedding_dictionary_e5.json \
  --prefix "passage: "
```

PR #21 가 multilingual-e5-large → KoE5 (query prefix), PR #22 가 query → passage prefix 로 교체 (과일 cross-check 정상화).

### 2.2 옛 흐름 — `src/E5_embedding_ver2.py`

multilingual-e5-large + mean-center + scaling (TOP1000=0.63 target). PR #21 이후 KoE5 raw + game.py 동적 sim_alpha 가 더 안정적이라 사용 빈도 줄어듦. 단 fallback scaling 로직은 [`tools/embedding_eval/apply_scaling_and_swap.py`](../../tools/embedding_eval/apply_scaling_and_swap.py) 에서 재사용.

### 2.3 모델 비교 / 교체 사이클

| 도구 | 역할 |
|---|---|
| `build_alt_embeddings.py` | 임의 모델 임베딩 생성 (KoE5 / KURE / 그 외) |
| [`tools/embedding_eval/build_hybrid_embedding.py`](../../tools/embedding_eval/build_hybrid_embedding.py) | 두 임베딩 결합 (concat / weighted average) |
| [`tools/embedding_eval/apply_scaling_and_swap.py`](../../tools/embedding_eval/apply_scaling_and_swap.py) | scaling 적용 (선택) + 자동 백업 후 메인 사전 swap |

### 2.4 spot-check ipynb

| ipynb | 역할 | PR |
|---|---|---|
| `notebooks/embedding_quality_exploration.ipynb` | 단일 모델 정합성 탐색 — "사과" top N, 과일 cross-check, 분포 | #20 |
| `notebooks/embedding_model_comparison.ipynb` | 두~세 모델 side-by-side 비교 | #21 |
| `notebooks/embedding_tuning_comparison.ipynb` | N-way 비교 (prefix 변경 + hybrid 포함) | #22 |

`notebooks/findings_YYYY-MM-DD_*.md` 패턴으로 발견 누적.

### 2.5 임베딩 모델별 spot-check 핵심 결과

| 모델 | 과일 cross-check (정답 "사과") | 채택 |
|---|---|---|
| multilingual-e5-large (옛 default) | bad — 글자 패턴 cluster ("-과" 끝 단어) | PR #21 전까지 사용 |
| KoE5 (query: prefix) | 개선 — 의미 단어 등장 | PR #21 채택 |
| **KoE5 (passage: prefix)** | **best** — top 100 안 과일 3/8 | **PR #22 채택 (현행)** |
| KURE-v1 | worst (동음이의어 분리로 단일 정답 게임엔 부적합) | 보류 — 미래 동음이의어 모드 후보 |
| Hybrid concat (KoE5+KURE) | 평균 — 결합 효과 미미 | 보류 |

---

## 3. 저장 형식 (PR #23)

### 3.1 NPZ (현행 default)

```bash
python tools/storage/convert_to_npz.py
# → data/embedding_dictionary_e5.npz (~235MB, 약 82% 절약)
```

- `np.savez(path, words=object_array, vectors=float32_matrix)` 한 파일
- `--compress` 옵션 (~150MB, load 1-2초 추가)
- 정확도 손실 0 (float32 그대로)

### 3.2 자동 분기 로드

[`src/core/embeddings.py`](../../src/core/embeddings.py) 가 `.json` / `.npz` 확장자 자동 감지:
- `.npz` → `np.load` + `words`/`vectors` 키 추출
- `.json` → orjson 파싱 (옛 형식 backward compat)

[`src/app.py`](../../src/app.py) 의 `_default_embedding_file()` 가 `.npz` 우선 → `.json` fallback.

---

## 4. HF Hub 배포

운영자 (`leo4study`) 가 새 사전 외부 배포:

```bash
hf upload leo4study/ccomantle-embeddings \
  ./data/embedding_dictionary_e5.npz --repo-type dataset
```

도커 / 다른 사용자 환경에서 `.env` 의 `EMBEDDING_HF_FILE=embedding_dictionary_e5.npz` 명시하면 새 사전 자동 다운로드.

---

## 5. 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `FASTTEXT_VEC_PATH` | (없음) | 단어 추출 시 .vec 파일 경로 |
| `EMBEDDING_HF_REPO` | (없음) | HF Hub repo (예: `leo4study/ccomantle-embeddings`) |
| `EMBEDDING_HF_FILE` | `embedding_dictionary_e5.npz` | HF 내 파일명 |
| `EMBEDDING_HF_TYPE` | `dataset` | HF repo 종류 |
| `EMBEDDING_FILE` | 자동 (`.npz` 우선) | 로컬 파일 경로 직접 지정 |

---

## 6. 관련 도구·문서

### 도구
- [`tools/dict_quality/`](../../tools/dict_quality/) — 사전 진단·보강
- [`tools/embedding_eval/`](../../tools/embedding_eval/) — 임베딩 비교·교체
- [`tools/storage/`](../../tools/storage/) — 저장 형식 변환
- [`src/make_words_from_vec.py`](../../src/make_words_from_vec.py) — FastText 추출
- [`src/E5_embedding_ver2.py`](../../src/E5_embedding_ver2.py) — multilingual-e5 + scaling (옛 default)

### 문서
- [`05_evaluation_methodology.md`](05_evaluation_methodology.md) — 평가 방법론 / Anna 정렬 트랙
- [`docs/CHANGELOG.md`](../CHANGELOG.md) — PR 별 진행 history
- `notebooks/findings_*.md` — 임베딩 spot-check 발견 누적
