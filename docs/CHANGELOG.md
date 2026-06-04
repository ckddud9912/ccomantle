# Refactoring Changelog

## [23번] docs 갱신 — KoE5 passage + NPZ 형식 + 새 도구 사이클 반영 (2026-06-04)

### 배경
PR #20-#22 의 임베딩 모델 교체·튜닝·저장 형식 변환 후 README + features/04 + .env.example 이 옛 정보 (multilingual-e5 / .json) 기준이라 갱신 필요.

### 변경 내용

**1. `README.md` — 5곳 갱신**
- 주요 기능·빠른 시작·기술 스택 표·환경변수 표 — KoE5 / NPZ 반영
- 임베딩 직접 생성하기 § 재구성 — Path A (FastText 부터) + Path B (기존 사전 보강) + 모델 비교/교체 흐름
- 사전 품질 진행 상황 §는 PR #15-#23 누적 박제 (60k / KoE5 passage / NPZ 82% 절약 / 과일 top 4)

**2. `docs/features/04_preprocessing.md` — 전면 재구성**
- 옛: 1회성 추출 + e5 raw/scaled JSON 흐름
- 새: 4-단계 사이클 (단어 추출 → 임베딩 → 저장 → HF 배포). 각 단계의 진단·교체·튜닝 도구 표
- 모델 비교 결과 표 (multilingual-e5 / KoE5 (query/passage) / KURE / Hybrid)
- spot-check ipynb 카탈로그 + findings_*.md 패턴 안내

**3. `.env.example` — `EMBEDDING_HF_FILE` default 가 .npz 라는 점 명시 (사용자 갱신본)**

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `README.md` | KoE5 passage + NPZ 반영, 임베딩 생성 § 재구성 |
| `docs/features/04_preprocessing.md` | 전면 재구성 (도구 사이클 반영) |
| `.env.example` | `.npz` default 명시 |
| `docs/CHANGELOG.md` | 본 항목 |

### API 변경
없음. 문서만.

---

## [22번] 임베딩 저장 format JSON → NPZ (1.3GB → 235MB, 82% 절약) (2026-06-03)

### 배경
PR #15-#22 진행하면서 임베딩 사전이 점차 누적 (multiple 모델·prefix·hybrid 실험). 각 사전 1.3GB JSON × 다수 → 디스크 용량 부족 발생. 본질 원인: JSON 의 number text encoding 비효율 (각 float 20+ chars). binary float32 로 가면 동일 정보, 약 80% 절약.

### 변경 내용

**1. `src/core/embeddings.py` — 확장자 분기 로딩**
- 옛: JSON 만 지원
- 새: `.npz` / `.json` 자동 분기. backward compat — 옛 JSON 그대로 작동
- `_load_json` / `_load_npz` 두 path, `_normalize_and_pack` 공통

**2. `src/app.py` — `.npz` 우선 자동 감지**
- `EMBEDDING_FILE` env 미설정 시 `_default_embedding_file()` 호출
- `data/embedding_dictionary_e5.npz` 있으면 우선, 없으면 `.json` fallback
- 옛 JSON 만 있는 환경도 그대로 작동

**3. `tools/storage/convert_to_npz.py` (신규)**
- JSON 1.3GB → NPZ ~235MB 변환 (또는 `--compress` 옵션으로 ~150MB)
- `np.savez(words=object_array, vectors=float32_matrix)` 한 파일
- 사용 안내·재업로드 명령 같이 출력

### 실측 결과
```
source: data/embedding_dictionary_e5.json (1302.5 MB)
[1/3] JSON 로드 + 파싱 — 60,000 단어, dim 1024, matrix 234.4 MB (raw float32)
[2/3] NPZ 저장
[3/3] 결과: data/embedding_dictionary_e5.npz (235.5 MB)
       절약: 1067.0 MB (81.9% ↓)
```

게임 로드 검증:
```
[INFO] Loading embeddings from .../embedding_dictionary_e5.npz...
[INFO] Loaded embeddings (60000 words). Ready.
```
→ 자동 감지 + 로드 성공. 게임 점수는 동일 (vector data 그대로).

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `src/core/embeddings.py` | `.json` / `.npz` 분기 로드 |
| `src/app.py` | `.npz` 우선 자동 감지 |
| `tools/storage/convert_to_npz.py` (신규) | 변환 도구 |
| `docs/CHANGELOG.md` | 본 항목 |

### API 변경
없음. 동작 호환.

### 다음 단계 (운영, PR 머지 후)
```bash
# 옛 JSON 삭제 (추가 1.3GB 회수)
rm data/embedding_dictionary_e5.json

# HF 재업로드 (.npz 외부 배포)
hf upload leo4study/ccomantle-embeddings \
  ./data/embedding_dictionary_e5.npz --repo-type dataset

# 도커/다른 사용자 환경에서 .npz 사용하려면 env 갱신:
# EMBEDDING_HF_FILE=embedding_dictionary_e5.npz
```

### 효과 summary
- 디스크: 1.3GB → 235MB per 사전 (**82% 절약**)
- HF 다운로드: 1.37GB → ~250MB (**약 80% 빠름**)
- 게임 로드 시간: 거의 동일 (JSON parse 대비 binary load 가 약간 더 빠를 수도)
- 정확도 손실: 0 (float32 그대로)

---

## [21번] 임베딩 튜닝 — KoE5 prefix 변경 (query: → passage:) + 도구 추가 (2026-06-03)

### 배경
PR #21 의 KoE5 교체 후 추가 개선 시도 — prefix 변경 (query/passage/없음) + KoE5+KURE hybrid concat. 비교 결과 **KoE5 (passage:) 가 과일 cross-check 압도적 best** (top 100 안 3/8 vs 다른 모든 모델 0/8) → swap 진행.

### 변경 내용

**1. `tools/embedding_eval/build_hybrid_embedding.py` (신규)**
- 두 임베딩 사전 → hybrid (concat 또는 weighted average)
- 각각 L2 normalize → 결합 → 다시 L2 normalize
- 미래 다른 임베딩 결합 실험에도 재사용 가능

**2. `notebooks/embedding_tuning_comparison.ipynb` (신규)**
- 임의 모델 N개 side-by-side 비교 scaffold
- §3 결정적 pair / §4 top N / §5 과일 cross-check / §6 강아지·학교 회귀 / §7 결정
- 비교 미실행 (미래 사용 대비 도구로 박아둠)

### 비교 결과 (`embedding_tuning_comparison.ipynb` 실측)

**과일 cross-check** (정답 "사과") — top 100 안 과일 개수:

| 모델 | top 100 / 8 | 평균 rank | 결정 |
|---|---:|---:|---|
| KoE5 (query:) — 옛 PR #21 | 0/8 | 802 | 옛 |
| **KoE5 (passage:)** ★ | **3/8** | 1,201 | **선정** |
| KoE5 (no prefix) | 0/8 | 3,000 | X |
| KURE | 0/8 | 22,946 | X (동음이의어 분리 부작용) |
| Hybrid concat (KoE5+KURE) | 0/8 | 2,975 | X |

**KoE5 (passage:) 의 과일 rank**: 수박 30위 (옛 244위) / 레몬 43위 (옛 453위) / 바나나 66위 (옛 118위) / 망고 131위 (옛 472위) / 딸기 217위 (옛 563위) / 포도 264위 (옛 874위) / 복숭아 269위 (옛 1,099위). **배만 8,588위 (옛 2,590위) 로 악화** — specific case, 게임 영향 미미.

### 사전 swap — KoE5 (query:) → KoE5 (passage:)

- 백업: `embedding_dictionary_e5.backup_query_*.json` (옛 query prefix 사전)
- 메인: `embedding_dictionary_e5.json` 을 passage prefix 로 교체
- game.py 의 동적 `sim_alpha` 분포 흡수 (PR #17 검증 패턴)

### 게임 테스트 실측 (정답 "사과")
- `sim_top1000_raw=0.6890, SIM_ALPHA=1.2402` 자연 분포
- top: 과일 0.768 / 수박 0.725 / 레몬 0.712 / 바나나 0.699 / 파랑 0.610 / 빨강 0.605
- **과일 4개 top 4 차지** — 사용자 통증 ("과일 점수 안 나옴") 완전 해소
- 미세 fail: 빨강 < 파랑 (0.005 차) — 임베딩 specific case 한계, 게임 진행 영향 X

### 사전 변경 영역
- 메인 사전: KoE5 query → KoE5 passage (60k, dim=1024)
- 임시 임베딩 사전들 (koe5_noprefix·hybrid_concat·kure 등) 은 작업 후 삭제 (디스크 5-6GB 회수)

### 다음 PR
**★ `feat/storage-binary-format`** — JSON 1.3GB × 다수 → npy ~250MB (80% 절약) binary 형식 변환. `src/core/embeddings.py` 의 로드 코드 수정 필요. 디스크 관리 측면 큰 효과.

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `tools/embedding_eval/build_hybrid_embedding.py` (신규) | hybrid 임베딩 생성 도구 |
| `notebooks/embedding_tuning_comparison.ipynb` (신규) | 모델 N-way 비교 scaffold |
| `docs/CHANGELOG.md` | 본 항목 |

### API 변경
없음. 도구만 추가.

---

## [20번] 임베딩 모델 교체 — multilingual-e5-large → KoE5 (한국어 의미 cluster 정상화) (2026-06-02)

### 배경
PR #19 (`findings_2026-06-02.md`) 의 결정적 발견 — multilingual-e5-large 가 한국어 의미보다 글자 패턴 cluster — 의 본질 해결. 같은 60k 단어에 한국어 특화 두 모델 (KoE5, KURE) 임베딩 재생성 → spot-check 비교 → 게임에 적합한 모델 선정.

### 변경 내용

**1. `tools/embedding_eval/build_alt_embeddings.py` (신규)**
- 임의 HuggingFace 임베딩 모델로 ccomantle 60k 단어 임베딩 재생성
- `--model`, `--output`, `--prefix` 옵션 (raw, L2 norm 만, scaling X)
- MPS 자동 가속 — KoE5 3분 8초, KURE 2분 24초

**2. `notebooks/embedding_model_comparison.ipynb` (신규)**
- 두~세 모델 side-by-side 비교
- §3 결정적 pair / §4 top N 비교 / §5 카테고리 cross-check

**3. `notebooks/findings_2026-06-02_3model_comparison.md` (신규)**
- 3-모델 비교 분석 + 결정 근거

**4. `tools/embedding_eval/apply_scaling_and_swap.py` (신규)**
- 대안 raw 임베딩에 scaling 적용 + 메인 사전 swap (백업 자동)
- 단 본 PR 의 KoE5 case 에선 scaling 실패 (SCALE_RANGE 최소값으로 떨어짐, diff 0.43) → scaling 건너뛰고 raw 그대로 swap

**5. 메인 사전 swap — multilingual-e5-large → KoE5 raw**
- 백업: `data/embedding_dictionary_e5.backup_swap_*.json`
- 메인: `data/embedding_dictionary_e5.json` 을 KoE5 raw 로 교체
- game.py 의 동적 `sim_alpha` 가 raw 분포 자동 흡수 (PR #17 의 검증된 패턴)

### 모델 비교 결과 요약

| 측면 | e5 (옛) | KoE5 (현 선정) | KURE (보류) |
|---|---|---|---|
| 글자 패턴 cluster | ⚠️ 심함 | 일부 잔존 | 사라짐 |
| **과일 cross-check** (정답 "사과") | bad | **best** | worst |
| 사과↔자동차 specific fail | ⚠️ | ⚠️ | ✓ 풀림 |
| 동음이의어 처리 | 한 의미 | 한 의미 | 다중 의미 (게임엔 ⚠️) |

**KURE 보류 이유**: 의미적으로는 가장 정확하지만 "사과 → 용서·죄송" 처럼 동음이의어를 다 cluster 해서 단일 정답 게임에 부적합. 과일 cross-check rank 가 e5 보다도 나쁨 (배 17,708위 vs e5 3,888위).

### 게임 테스트 실측 (정답 "사과")

- `sim_top1000_raw=0.6653, SIM_ALPHA=1.1338` (e5: 0.2053 / 0.2918 → 분포가 자연스러워짐)
- top 6: 수박 0.664 / 딸기 0.643 / 포도 0.633 / 배 0.608 / 끝 0.561 / 꿈 0.521
- **과일 4개가 top 4 차지** — e5 에선 배만 1위 (다른 과일들 rank 1000+ 밖) → 정상화
- "끝"·"꿈" 같은 옛 추가 단어도 정상 작동

### 파인튜닝 결정
**X**. KoE5 가 사용자 통증 (의미 cluster) 의 80% 해결. 남은 fail (사과↔자동차 같은 edge case) 은 multilingual-e5 base 의 본질 한계 — fine-tuning 으로도 보장 안 됨. 시간 (며칠~주) 대비 효과 작음. 대신 `feat/sim-calibration` / `feat/game-log-rejected` 같은 다른 트랙으로 보완.

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `tools/embedding_eval/build_alt_embeddings.py` (신규) | 임의 모델 임베딩 생성 |
| `tools/embedding_eval/apply_scaling_and_swap.py` (신규) | scaling + swap 도구 (이번 case 엔 scaling 실패, swap 만 활용) |
| `notebooks/embedding_model_comparison.ipynb` (신규) | 3-way 비교 ipynb |
| `notebooks/findings_2026-06-02_3model_comparison.md` (신규) | 비교 결과 + 결정 근거 |
| `docs/CHANGELOG.md` | 본 항목 |

### API 변경
없음. 임베딩 사전 (gitignored) 만 변동. game.py 의 `sim_alpha` 동적 보정이 모델 교체 흡수.

### 다음 단계 (운영자, PR 머지 후)
HF Hub 재업로드:
```bash
hf upload leo4study/ccomantle-embeddings \
  ./data/embedding_dictionary_e5.json --repo-type dataset
```

### 미래 트랙 (선택)
- `feat/sim-calibration` — 남은 specific fail 영향 줄이는 점수 보정 강화
- `feat/dict-cleanup-activations` — 활용형 (조사) 정리. KoE5 의 top 차지 패턴 일부 해소
- (장기) KURE 적합 시나리오 — 동음이의어 명확히 처리하는 게임 모드 추가 시 후보

---

## [19번] 임베딩 의미 정합성 탐색 — ipynb scaffold + 결정적 발견 (2026-06-02)

### 배경
PR #15~#18 의 사전 어휘 보강이 사용자 통증 일부 해소했지만, 사용자가 정작 본질로 짚은 것은 **"임베딩 유사도 점수가 의미적으로 이해 안 됨"** (정답 "사과" 에 과일이 안 가깝게 나옴 등). 이건 사전 크기 문제 아닌 **임베딩 모델 자체의 한국어 의미 처리 품질** 영역. 탐색 단계로 ipynb 도입.

### 변경 내용

**1. 신규 [`notebooks/embedding_quality_exploration.ipynb`](../notebooks/embedding_quality_exploration.ipynb)**
- 7개 섹션 scaffold: 임베딩 로드 / helper / 단일 정답 top N / 다양한 정답 일괄 비교 / 카테고리 cross-check / 분포 시각화 / 종합 발견
- `ANSWER` 변수 바꿔가며 spot-check 하는 탐색 결 도구
- 향후 KoE5/KURE 등 다른 임베딩 비교 시 그대로 재사용

**2. 신규 [`notebooks/findings_2026-06-02.md`](../notebooks/findings_2026-06-02.md)**
- 첫 실행 결과 발견 5가지 정리 + 근본 원인 진단 + 다음 PR 결정
- 향후 동일 탐색 반복 시 `findings_YYYY-MM-DD.md` 누적

### 결정적 발견

1. **임베딩이 의미보다 글자 패턴에 강하게 반응** — 정답 "사과" top 50 의 다수가 "-과" 끝 단어 (제과·인과·여과·안과·내과·외과 등). 의미 무관한데 마지막 글자 같으면 cosine 높음
2. **자동차가 배보다 사과에 더 가깝다** — `사과↔배=0.1239`, `사과↔자동차=0.1668`. 의미 정합성 깨짐
3. **과일 cluster 작동 X** — 정답 "사과" 에서 배 3,888위, 포도 737위, 딸기 2,316위 등. 도메인 cluster 완전 fail
4. **단어별 분포 들쭉날쭉** — "강아지·학교·사랑" 은 의미 cluster 정상 작동 (고양이·스쿨·로맨스 등 잘 묶임). 반면 "사과·배·차·집" 은 글자 패턴 cluster. **같은 모델 일관성 없음**
5. **활용형이 top 차지** — 조사 활용형 ("사과와"·"사과는") 이 top 5 안

→ 본질 원인: **multilingual-e5-large 가 한국어 의미 처리에 약함**. 한국어 특화 모델 (KoE5/KURE) 필요.

### 다음 PR

**★ `feat/embedding-model-comparison`** (강력 추천) — 본질 해결 트랙
- 같은 60k 단어에 KoE5 임베딩 재생성 → ipynb 동일 spot-check
- KURE-v1 추가 검토
- 한국어 특화 모델이 글자 패턴 cluster 해소하는지 검증

보류 트랙:
- `feat/dict-cleanup-activations` — 표면적 효과만, 모델 교체 후 검토
- `feat/sim-calibration` — distribution 자체는 정상 (mean=0.0040)

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `notebooks/embedding_quality_exploration.ipynb` (신규) | 탐색 도구 scaffold |
| `notebooks/findings_2026-06-02.md` (신규) | 첫 발견 누적 doc |
| `docs/CHANGELOG.md` | 본 항목 |

### API 변경
없음. 탐색 단계, 코드/사전 변경 X.

---

## [18번] 우리말샘·ko 위키 diff source 추가 + 우리말샘 명사 5,000 보강 (55k → 60k) (2026-06-02)

### 배경
[17번] 의 mecab-ko-dic (2018, 8년 묵음) 기반 보강이 사용자 진짜 통증 (동시대 일반 명사 누락) 과 결이 안 맞다는 push-back. 진짜 한국어 사전 source 를 도입해서 reference 다양화 + 누락 진단·보강 인프라 확장.

### 변경 내용

**1. 신규 — `tools/dict_quality/diff_kowiki.py`**
- 한국어 위키피디아 latest titles dump (ns0 articles) 자동 다운로드 (~8MB gz)
- 한글-only / 길이 1-6 필터 → 492,561 titles
- mecab NNG cost cross-ref (이전 PR 다운로드 재사용)
- ccomantle 사전과 diff → 누락 분류 (mecab+kowiki 교집합 vs ko 위키 only)
- 산출: `data/quality_report_kowiki.json` + `data/missing_words_candidates_kowiki.json`
- **결과**: coverage 3.65%, 누락 474k. 단 "ko 위키 only" 435k 의 대부분은 일본 지명/인명 (가가린·가가미이시·가가와현 등) — 사전 추가 부적합. **"mecab+kowiki 교집합 39k" 만 유의미**

**2. 신규 — `tools/dict_quality/diff_urimalsaem.py`**
- 국립국어원 우리말샘 bulk dump (25 chunk × 50k = ~1.2M entry) 처리
- 핵심 필터: `senseinfo.type == "일반어"` (방언·옛말·북한어 제외) + 표제어 하이픈 제거 ("겁-쟁이" → "겁쟁이")
- POS 별 set 분류 (명사·동사·형용사·부사·기타)
- mecab NNG cost cross-ref (명사만)
- 산출: MD 리포트 (사용자 읽기용) + JSON (머신 읽기용) + POS 별 후보 4개 파일
- **결과**: 명사 318k 중 ccomantle 교집합 18,680 → coverage 5.87%. mecab+우리말샘 교집합 명사 171,829

**3. `tools/dict_quality/expand_dict.py` 옵션화**
- 옛: `data/missing_words_candidates.json` 하드코딩
- 새: `--source <path>` + `--filter-source <value>` 옵션
- 우리말샘·ko 위키·mecab 등 임의 진단 산출 받아 보강 가능
- 호환: `--source` 미지정 시 옛 path default

**4. 우리말샘 명사 5,000 보강 실행 → 사전 55k → 60k**
- 명령: `expand_dict.py --source data/missing_words_urimalsaem_nouns.json --filter-source urimalsaem+mecab --top-n 5000`
- 5,000개 다 신규 (기존 ccomantle 60k 와 중복 0)
- distribution 보존 (PR #17 의 검증된 방식)
- 게임 테스트 통과: `sim_top1000_raw=0.2053, SIM_ALPHA=0.2918` (PR #17 와 동일 — 분포 완벽 보존)
  - 새 단어 "복면" → 0.419 (정답 "사과"). 합리적
  - 옛 단어 "배" 0.544, "끝" 0.441 — 분포 완전 보존

### 결정한 정책

- **명사 only** 유지 (사용자 결정): 동사·형용사는 추가 X (활용형으로 인한 유사 순위 정렬 어려움 회피)
- **우리말샘 only 후보는 추가 X**: top 50 보면 가가린/가가미이시/가가와현 같이 일본 지명·전문용어 다수
- **mecab+우리말샘 교집합** 만 = 양쪽 다 인정된 안전한 어휘

### 알려진 한계 — 다음 PR 후보

mecab cost 가 너무 균질 (top 5,000 = cost 2628-2639) 해서 **실제 사용 빈도와 매칭이 정확하지 않음**. 결과:
- "자두" (mecab cost 2,665) 가 우리말샘+mecab 정렬에서 **156,099번째** 위치 → top 5,000 으로는 못 잡힘
- 흔한 과일 중 자두·귤·참외 여전히 누락
- "복면" (cost 2628) 이 "자두" 보다 흔한가? **No**. mecab cost = 빈도 아님

→ 다음 PR 후보: **`feat/dict-frequency-from-kowiki-body`** — ko 위키 article body 빈도로 정렬. mecab cost 보다 실제 사용 빈도 정확. dump ~500MB, 처리 무거움.

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `tools/dict_quality/diff_kowiki.py` (신규) | ko 위키 titles diff |
| `tools/dict_quality/diff_urimalsaem.py` (신규) | 우리말샘 diff + POS 분류 + MD 리포트 |
| `tools/dict_quality/expand_dict.py` | `--source`/`--filter-source` 옵션 추가 |
| `docs/CHANGELOG.md` | 본 항목 |

### API 변경
없음. 사전 데이터 (`/data` gitignored) + 진단 도구만 변동.

### 다음 단계 (운영자, PR 머지 후)
HF Hub 재업로드:
```bash
hf upload leo4study/ccomantle-embeddings \
  ./data/embedding_dictionary_e5.json --repo-type dataset
```

---

## [17번] 사전 어휘 보강 — 누락 한국어 명사 5000개 추가 (50k → 55k) (2026-06-02)

### 배경
[15번] 의 진단 결과 누락 단어 5000개 후보 식별. [16번] 의 필터 수정만으로는 데이터 자체 (FastText 50k cap) 한계 해소 불가 → 직접 보강 트랙. FastText `.vec` 재다운로드 없이 ccomantle 사전에 누락 명사 추가.

### 변경 내용

**1. 신규 도구 — [`tools/dict_quality/expand_dict.py`](../tools/dict_quality/expand_dict.py)**
- `missing_words_candidates.json` 의 top N (default 5000) 단어 추출
- 기존 임베딩과 중복 제거
- 새 단어만 multilingual-e5-large 로 E5 임베딩 생성 (~17초, MPS)
- 기존 50k scaled 분포 보존 (mean-center/scale 재실행 X)
- 자동 백업 → `data/embedding_dictionary_e5.backup_*.json`
- `--dry-run` 옵션
- 의존성: `requirements-dev.txt` (torch + transformers + tqdm)

**2. coverage_check.py 수정** — 실제 게임 사전을 측정 기준으로
- 옛: `data/words_50000.json` (빌드 input) 을 봤음
- 새: `data/embedding_dictionary_e5.json` (게임이 실제 로드) 의 키를 봄
- expand 같은 작업으로 임베딩 JSON 만 갱신되면 진단도 자동 반영됨

**3. 실측 효과**
- **Coverage**: 4.81% → 7.25% (교집합 9,876 → 14,876)
- **누락**: 195,393 → 190,393 (정확히 5,000 감소)
- 게임 테스트 (정답 "사과", `sim_top1000_raw=0.2053`):
  - 옛 1글자 명사 "끝" → 0.441 (이전 입력 자체 거부됨)
  - 옛 게/히 끝 명사 "무게" → 0.430
  - 옛 흔한 명사 "큰일" → 0.327
  - "사전에 없는 단어" 에러 0건
  - 옛 단어 "배"·"주스"·"빨강" 의 점수 분포 보존됨

### 설계 결정 — distribution 보존 (mean-center/scale 재실행 안 함)

처음에는 머지 후 `scale_embeddings` 재실행을 시도했으나 점수가 모두 0 으로 망가짐. 원인:
- 기존 50k 가 이미 mean-centered + scaled 임베딩
- mean-centering 한 번 더 적용 → 기존 단어들의 cosine 분포 압축 → 다수가 음수/0 근처
- [game.py:18-23](../src/core/game.py#L18-L23) 의 `_scale_scalar` 가 `x <= 0` 이면 0 반환

해결: scale 재실행 step 제거. 기존 50k scaled 그대로 + 새 5k 는 E5 raw (unit-norm) 그대로 추가. game.py 의 동적 `sim_alpha` (매 정답마다 1000위=0.63 으로 재계산) 가 distribution 차이를 흡수 → 옛 단어 점수 보존 + 새 단어도 합리적 점수.

### 다음 단계 — HF Hub 업로드 (운영자 작업)
현재 git 추적되는 코드만 PR 에 포함. 실제 임베딩 JSON 은 `/data` 가 gitignored 라 운영자가 HF 에 업로드:
```bash
huggingface-cli upload leo4study/ccomantle-embeddings \
  ./data/embedding_dictionary_e5.json --repo-type dataset
```
이게 끝나면 도커/다른 사용자도 새 사전 사용 가능.

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `tools/dict_quality/expand_dict.py` (신규) | 어휘 보강 도구 |
| `tools/dict_quality/coverage_check.py` | 실제 게임 사전(`embedding_dictionary_e5.json`) 을 측정 기준으로 |
| `docs/CHANGELOG.md` | 본 항목 |

### API 변경
없음. 사전 데이터 (`/data` 하위) 만 변동. game.py / endpoints 변경 X.

---

## [16번] 사전 추출 필터 결함 수정 — 1글자 명사 허용 + 광범위 부사 필터 제거 (2026-06-01)

### 배경
[15번] 의 진단 결과 ([`docs/features/05_evaluation_methodology.md`](features/05_evaluation_methodology.md#L188) §1.7) finding 1·2 에서 발견된 `is_valid_word` 의 두 결함을 직접 수정.

### 변경 내용

**[`src/make_words_from_vec.py`](../src/make_words_from_vec.py) 의 `is_valid_word` 두 조건 변경**:

1. **길이 cutoff 완화**: `2 <= len <= 6` → `1 <= len <= 6`
   - finding 1: "끝"·"꿈"·"눈"·"값"·"봄"·"날"·"칸"·"쌀" 등 흔한 1글자 명사 약 990개 누락 (mecab NNG 기준)
   - 게임에서 흔하게 떠올리는 1글자 명사가 거부되는 통증 해소

2. **`ADVERB_PATTERN` 제거**: 옛 `r".+(게|히)$"` 가 너무 광범위
   - finding 2: 명사 "가게"·"무게"·"모기"·"공기"·"인사" 등 약 314개를 부사로 오인 거부
   - 진짜 부사 ("빠르게"·"조용히") 는 FastText 빈도 cap 50k 자체로 대부분 거름. 일부 들어와도 사전 오염 미미
   - 더 정밀한 필터 (mecab NNG 화이트리스트) 는 src 의존성 부담으로 보류. 필요 시 다음 PR

### 효과 발휘 조건
본 변경은 **사전 재추출 시 효과 발휘**. 이미 생성된 `data/words_50000.json` / `data/embedding_dictionary_e5.json` 자체는 변동 없음. 효과 보려면:

```bash
# FastText .vec 가 있는 환경에서
export FASTTEXT_VEC_PATH=/path/to/cc.ko.300.vec
python src/make_words_from_vec.py     # 새 필터로 50,000 단어 재추출
python src/E5_embedding_ver2.py        # 임베딩 재생성
# 그 후 data/embedding_dictionary_e5.json 을 HF dataset 에 재업로드
```

→ 추출 + 임베딩 재생성 + HF 업로드는 운영 사이드 작업. 환경 갖춰지면 진행.

### 측정 (다음)
사전 재생성 후 [`tools/dict_quality/coverage_check.py`](../tools/dict_quality/coverage_check.py) 재실행 → coverage 4.81% 대비 변화 확인. [15번] §1.7 의 baseline 과 비교.

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `src/make_words_from_vec.py` | `is_valid_word` 두 조건 (3줄 줄음) + 변경 의도 주석 |
| `docs/CHANGELOG.md` | 본 항목 |

### API 변경
없음. 추출 도구의 필터만 변경.

---

## [15번] 사전 품질 진단 — 누락 단어 탐지 인프라 + 평가 방법론 doc (2026-06-01)

### 배경
사용자가 게임 플레이 중 "사전에 없는 단어" 거부를 자주 겪는다는 통증 보고. (주)애나 정렬 평가 트랙의 첫 작업으로, ccomantle 임베딩 사전의 coverage 를 정량 진단하고 어떤 한국어 명사가 누락됐는지 식별하는 인프라 구축.

### 변경 내용

**1. 신규 평가 방법론 doc — [`docs/features/05_evaluation_methodology.md`](features/05_evaluation_methodology.md)**
- §0 현재 평가 / 점수 계산 흐름 (코드 위치 명시: `make_words_from_vec.py` 필터 / `game.py` sim_alpha / `embeddings.py`)
- **§1 누락 단어 탐지 ★** (본 PR 본진)
  - 문제 정의 + 4 가설 (cap / 활용형 점유 / 필터 과잉 / 외래어 누락)
  - reference dict 5 후보 비교 → **mecab-ko-dic NNG.csv raw 단독** 결정 (의존성 0, POS 분리 무료, 향후 확장 가능)
  - NNG.csv 행 구조 + KoNLPy 와의 관계 설명
  - 측정 지표 4종 (coverage / missing count / top-K / filter breakdown)
  - 우선순위 산정 공식
  - 산출물 명세
  - 실측 결과 (§1.7)
  - 다음 단계 3 트랙 (필터 수정 / 어휘 보강 / 게임 로그)
- §2-5 placeholder (활용형 클러스터 / intrinsic eval / sim 보정 / 게임 로그)

**2. 신규 진단 도구 — [`tools/dict_quality/coverage_check.py`](../tools/dict_quality/coverage_check.py)**
- mecab-ko-dic 2.1.1-20180720 자동 다운로드 (47.5MB, 첫 실행만)
- `NNG.csv` 파싱 (UTF-8 + EUC-KR 폴백)
- ccomantle 사전과 비교 + 필터 거부 원인 분류
- 산출: `data/quality_report.json` + `data/missing_words_candidates.json` (top 5000)
- 의존성 0 (urllib + tarfile + json 표준 라이브러리만)

**3. 결정적 발견 (실측)**
- Coverage 4.81% (mecab NNG 205,269 중 9,876 만 ccomantle 에 있음)
- **finding 1**: 1글자 명사 990개 누락 ("끝"·"꿈"·"봄"·"눈"·"값" 등 흔한 명사) — 현 `is_valid_word` 의 길이≥2 제약 때문
- **finding 2**: "게/히" 부사 필터 (`ADVERB_PATTERN`) 가 명사도 같이 잡음 — "가게"·"무게"·"모기" 등 314개
- **finding 3**: 진짜 흔한 명사 "학년도"·"큰일" 도 누락 — FastText 50k cap 또는 단어 분리 이슈
- 압도적 다수 누락 (98.4%) 은 cap/FastText 부재 원인

**4. .gitignore 갱신**
- `tools/dict_quality/refs/` 추가 (다운로드한 mecab tarball + 압축 해제분)

### 다음 PR 권장 순서
1. **`feat/dict-filter-fix`** (가장 작은, 가장 큰 효과 추정) — finding 1·2 직접 해소 (길이≥1, 부사 필터 정밀화 또는 제거)
2. **`feat/dict-expand-coverage`** — `missing_words_candidates.json` 기반 어휘 보강 + cap 확장
3. **`feat/dict-game-log-rejected`** (선택) — 게임 로그 기반 실측 검증

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `docs/features/05_evaluation_methodology.md` (신규, 274+ 줄) | 평가 방법론 + §1 실측 결과 |
| `tools/dict_quality/coverage_check.py` (신규) | 진단 도구 |
| `.gitignore` | refs/ 캐시 제외 |
| `docs/CHANGELOG.md` | 본 항목 |

### API 변경
없음. 진단 단계, 코드/엔드포인트 변경 X.

---

## [14번] 라운드 진행 시 참여 팀 누락 검사 (2026-05-10)

### 배경
어드민이 라운드를 진행할 때, 1라운드 베이스라인 팀 중 현재 라운드에서 아직 제출하지 않은 팀이 있으면 모르고 그냥 다음 라운드로 넘어가버리는 경우가 있었음. 운영 중 사람 손으로 진행하는 흐름이라 사고 방지용 가드 한 단계 필요.

### 변경 내용

**프론트엔드만 — `changeRound` 에 사전 검사 추가**
- 어드민이 라운드 버튼을 누르면 `/leaderboard` 한 번 fetch 해서 현재 상태 확인
- **앞으로 진행하는 경우만** 검사 (round > currentRound). 뒤로 가거나 같은 라운드는 검사 없음
- 1라운드 참여 팀(베이스라인) 중 현재(떠나는) 라운드에 빠진 팀이 있으면 confirm
- confirm 메시지에 **빠진 팀명을 명시** → 어드민이 누구인지 즉시 파악
  ```
  2라운드에 참여하지 않은 팀: B
  (1라운드 기준 2팀 중 1팀만 참여)

  모든 팀이 참여하지 않았습니다. 다음 라운드로 넘어가시겠습니까?
  ```
- 검사용 fetch 가 실패해도 라운드 변경 자체는 진행 — 어드민 의도가 검사보다 우선

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `static/js/admin.js` | `changeRound` 에 leaderboard 사전 fetch + 누락 팀 검사 |

### API 변경
없음. 기존 `/leaderboard` 데이터만 활용.

### 검사 동작 표
| 시나리오 | 동작 |
|---|---|
| R1 → R2, 모두 참여 | 무프롬프트 통과 |
| R1 → R2, 1팀 누락 | confirm (누락 팀명 표시) |
| R3 → R1 (뒤로) | 검사 없이 진행 |
| R2 → R2 (같은) | 검사 없이 진행 |
| 정답 미설정 / R1 비어있음 | 베이스라인 0 → 검사 스킵 |
| R1 → R5 (점프), 1라운드 비교용으로 자기 자신 | missing 없음 → 무프롬프트 |

---

## [13번] 게임 재시작 — 팀 정보 유지하고 라운드만 리셋 (2026-05-10)

### 배경
게임 종료 후 같은 팀들로 새 정답을 가지고 다시 플레이하려면 서버 재시작 외에 방법이 없었음. handoff.md "Pending TODO" 에 명시되진 않았지만 운영 시 자연스럽게 필요한 기능.

### 변경 내용

**백엔드 — `GameState.restart()` + `POST /restart`**
- 신규 메서드는 `team_colors` 만 유지하고 나머지(rounds, current_round, finished, answer_word, answer_vector, word_to_rank, sim_top*, sim_alpha) 모두 초기화
- 기존 `reset_for_answer` 는 그대로 — 이쪽은 정답 설정 시점에 team_colors 까지 비우는 동작 유지 (장기적으론 분리하는 게 깔끔할 수 있음)
- 락(`asyncio.Lock`) 안에서 처리해 진행 중 제출과 직렬화

**어드민 — "게임 재시작" 버튼 + 안전한 트리거**
- 경기 컨트롤 섹션에 버튼 1개 추가
- 클릭 → confirm "게임을 재시작하시겠습니까? (라운드와 정답은 초기화, 팀명·팀색은 유지)" → 예 → `/restart`
- 처음에 "경기 종료" 직후 자동으로 재시작 confirm 을 같이 띄우는 안도 검토했으나, 어드민의 misclick 으로 플레이어들의 final 결과 viewing 화면이 갑자기 끊길 수 있어서 **명시적 버튼만** 으로 한정
- 종료 후엔 alert 텍스트로 "재시작하려면 버튼을 누르세요" 안내만 첨부

**게임 클라이언트 — finished 상태에서도 폴링 유지**
- 기존: `data.finished=true` 보면 `clearInterval` 로 폴링 중단 → 어드민이 재시작해도 클라이언트가 알 길 없음
- 변경: `finalShown` 플래그 도입. finished=true 첫 진입 시에만 final 결과 로드, 이후엔 폴링은 계속 1.5s 마다 도는 상태
- finished true → false 전환 감지 시 final 오버레이 닫고 `lastCorrectKey` 리셋 → 빈 보드로 자동 복귀 (사용자가 새로고침할 필요 없음)

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `src/core/game.py` | `restart()` 메서드 추가 |
| `src/api/routes.py` | `POST /restart` 엔드포인트 |
| `static/admin.html` | "게임 재시작" 버튼 |
| `static/js/admin.js` | `restartGame()` + 버튼 핸들러. 종료 alert 텍스트 안내 |
| `static/js/game.js` | `finalShown` 플래그, 폴링 영속화, 재시작 자동 복귀 로직 |

### API 변경
- `POST /restart` 신규. 응답 `{"status": "ok"}`. 인증 없음 (어드민 페이지 자체가 신뢰 환경 가정 — handoff.md 의 보안 우선순위 참고)

### 확장 여지
재시작 후 자동으로 새 정답까지 받는 흐름 (예: 어드민에서 정답 입력 → 재시작 동시 처리) 은 별도 작업으로 분리 가능. 현재는 어드민이 재시작 → 새 정답 설정 두 단계.

---

## [12번] 내 팀 행 하이라이트 + 막대 색상 그라디언트 (2026-05-10)

### 배경
handoff.md "Pending TODO > 사용성" 두 항목 처리:
- 자기 팀 행 강조 (리더보드에서 내 행을 한눈에 찾기 어려움)
- 유사도에 따른 막대 색상 그라디언트 (현재 단색 파란색이라 진행감 약함)

함께 favicon 404 도 해결 (별도 PR 분리하지 않고 한 번에).

### 변경 내용

**1. 내 팀 행 하이라이트**
- 입력한 팀명을 `sessionStorage` 에 저장 (탭별 격리). 같은 브라우저 다른 탭에서 다른 팀이 흘러들지 않게 함 — 다중 디바이스 가정이 어려운 테스트 환경에서도 한 노트북에 여러 탭 띄우면 동작
- `localStorage` 는 새 탭 첫 진입 시 입력란 자동 채우기 폴백으로만 사용
- 팀 색상 picker 도 동일 패턴으로 sessionStorage 에 저장 → 새로고침 후에도 picker UI 가 마지막 선택색 유지
- "내 팀" 식별은 **이 탭에서 마지막으로 제출한 팀** 기준 (입력란 글자만 바꾸는 걸로는 안 바뀜) — 오타나 다른 탭 영향 차단
- CSS `.row-mine` : 노란 톤 배경 (`rgba(251,191,36,0.12)`) + 좌측 inset accent 막대

**2. 막대 색상 그라디언트**
- `barColorFor(row)` 함수: rank 1 → hue 130(초록), rank 1000 → hue 0(빨강) 으로 HSL 매핑
- `is_answer` → `var(--accent)` (정답 노란색)
- rank > 1000 또는 없음 → `var(--border-input)` (회색)
- bar-fill 의 width(rank 기반) 와 background(rank 기반) 가 같이 움직이며 진행감 강화

**3. favicon 404 해결**
- `static/index.html` 에 `<link rel="icon" href="data:,">` 추가 — 빈 data URL 로 브라우저의 favicon.ico 자동 요청을 흡수해서 서버 로그의 404 제거
- 함께 viewport meta 도 추가 (모바일 렌더링 일관성)

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `static/js/game.js` | sessionStorage 기반 내 팀 식별, `getMyTeam`, `barColorFor`, 색상 picker 복원 |
| `static/css/game.css` | `.row-mine` 스타일 |
| `static/index.html` | favicon 빈 data URL + viewport meta |

### API 변경 없음
모두 클라이언트 측 변경. 서버 `team_color` / `rank` 응답 필드 그대로 사용.

---

## [11번] .env 자동 로드 — 도커/비도커 풀 자동화 (2026-05-09)

### 배경
도커 컴포즈는 `.env`를 자동으로 환경변수로 주입하지만,
직접 `python src/app.py`로 실행하면 `.env`가 무시됨 → `EMBEDDING_HF_REPO` 같은 값을 매번 `export` 해야 했음.
"풀 자동화" 목표에 어긋나는 마지막 마찰점.

### 해결
- `requirements.txt`에 `python-dotenv` 추가
- `src/app.py` 상단에 `load_dotenv(.env)` 호출
- 이미 export된 환경변수가 있으면 그게 우선 (override=False, dotenv 기본 동작)

### 효과
| 실행 방식 | 변경 전 | 변경 후 |
|---|---|---|
| `docker compose up` | `.env` 자동 적용 ✓ | `.env` 자동 적용 ✓ |
| `python src/app.py` | `export ...` 필수 ✗ | `.env` 자동 적용 ✓ |

이제 두 경로 모두 `cp .env.example .env` 한 번만 하면 동일하게 동작.

### 변경된 파일
- `requirements.txt` : `python-dotenv` 추가
- `src/app.py` : 상단에 `load_dotenv` 호출 (try/except로 미설치 환경에서도 죽지 않게)
- `README.md` : 옵션 B 섹션의 export 안내 → "자동 로드됨" 으로 단순화

---

## [10번] 최종 결과 화면 — 단어 랭킹 + 팀 평균 두 섹션으로 분리 (2026-05-09)

### 배경
9번에서 최종 결과에 팀별 라운드별 단어를 칩 형태로 inline 표시했으나,
모든 라운드 단어가 한 줄에 다 나와서 가독성 떨어진다는 피드백.
"가장 점수 높은 단어"가 가장 궁금한 정보인데 묻혀버림.

### 변경 내용

**최종 결과 팝업을 두 섹션으로 분리**
- **🥇 단어 랭킹**: 모든 제출 단어를 점수순. 1위는 accent 색 강조 + 배경 하이라이트.
  자체 스크롤 (max-height: 36vh).
- **🏅 팀별 평균**: 각 팀의 평균 + 베스트 단어 한 개만 (라운드, 점수 함께).
  1위 팀 강조.

이로써 사용자는:
1. 최상단에서 "어떤 단어가 가장 가까웠는지" 즉시 확인
2. 그 아래에서 팀별 평균 + 각 팀이 낸 베스트 단어만 간결하게 확인
3. 모든 라운드 단어를 줄줄이 보지 않아도 됨

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `static/game.html` | final-card 안 두 섹션 신규 (final-words-tbody, final-teams-tbody) |
| `static/css/game.css` | `.final-section`, `.final-section-scroll`, `.row-top1` 스타일 추가. 옛 `.final-words` / `.fw-*` 제거 |
| `static/js/game.js` | `renderFinalWordRanking`, `renderFinalTeamRanking` 분리. `appendCell` / `appendTeamCell` 헬퍼 |

### API 변경 없음
서버 응답은 이미 9번에서 추가한 `submissions` 배열로 충분. 클라이언트 렌더링만 변경.

---

## [9번] 게임 레이아웃 3분할 + 최종 결과에 라운드별 단어 표시 (2026-05-09)

### 배경
기존 4분할(2x2 grid)에서 과거 라운드를 홀짝으로 나눠 좌우 패널에 분배.
가독성 안 좋다는 피드백 — 한 라운드 보려고 두 패널을 번갈아 보게 됨.
또 최종 결과 화면에 평균 유사도만 표시되어, 게임 끝나고 정작 "어떤 단어 냈는지"
정보가 빠져 있었음.

### 변경 내용

**레이아웃 — 4분할에서 3분할로**
- 좌상: 입력 패널 (팀 이름·색상·단어)
- 좌하: 현재 라운드 리더보드
- 우: 과거 라운드 통합 (전체 높이 차지, 최신이 위로 정렬)

CSS grid-template-areas 사용:
```
"input past"
"current past"
```

**최종 결과 — 라운드별 단어 표시**
- 서버: `final_result()`가 팀별 `submissions[]` 배열 반환 (라운드 + 단어 + 유사도)
- 클라이언트: 표 컬럼 추가 + 칩 형태로 R1[사과 0.91] R2[배 0.83] ... 표시

**부가 개선**
- 팀명 입력칸 `autofocus` (페이지 진입 후 바로 타이핑 가능)
- 라운드 진행 dots 6개 (●○○○○○ 형태로 시각화)
- 과거 라운드 시간 역순 (최신이 위)

### 변경된 파일
| 파일 | 내용 |
|---|---|
| `static/game.html` | div 클래스 grid-2x2 → layout-3pane, 패널 4 → 3, autofocus, 진행 dots, final 컬럼 |
| `static/css/game.css` | grid-template-areas 정의, .round-progress / .final-words 스타일, 모바일 분기 갱신 |
| `static/js/game.js` | renderPastRounds 단일 panel + 역순, renderRoundProgress 추가, loadFinalResult 라운드 단어 표시, escapeHtml 헬퍼 |
| `src/core/game.py` | final_result()가 팀별 submissions 배열 포함하도록 확장 |

### API 변경
**`/final_result` 응답 확장** (이전과 호환, 필드 추가만)
```json
{
  "answer": "...",
  "result": [
    {
      "team": "팀A",
      "team_color": "#...",
      "avg": 0.821,
      "submissions": [
        { "round": 1, "word": "사과", "similarity": 0.91 },
        ...
      ]
    }
  ]
}
```

---

## [7번] 1줄 실행 인프라 — docker compose + HF Hub 자동 다운로드 (2026-05-09)

### 배경
"git clone → docker compose up" 한 줄로 즉시 실행 가능해야 한다는 목표.
가장 큰 차단요소였던 임베딩 파일 배포(수백 MB JSON, git에 못 올림) 해결.

### 신규 파일
- `docker-compose.yml` — 단일 서비스, 포트 7860, data 볼륨 마운트, 헬스체크 포함
- `.env.example` — 환경변수 템플릿, 사용자가 cp 후 값 채우는 방식

### 변경된 파일
- `src/app.py`: `_try_hf_download()` 함수 추가. `EMBEDDING_HF_REPO` 환경변수 설정 시
  서버 기동 전에 HF Hub에서 임베딩 자동 다운로드 → `data/`에 캐싱
- `requirements.txt`: `huggingface_hub` 추가 (다운로드용)
- `.dockerignore`: `.env`, `.env.local`, `docker-compose.yml` 제외 (시크릿/dev 파일)
- `README.md`: 빈 메타데이터 페이지에서 → 운영자가 1회 읽고 그대로 실행 가능한 가이드

### 임베딩 파일 처리 — 하이브리드 전략
| 옵션 | 설정 | 동작 |
|---|---|---|
| A. HF Hub 자동 다운로드 (권장) | `.env`에 `EMBEDDING_HF_REPO=user/repo` | 첫 기동 시 자동 받음, `data/`에 캐싱 |
| B. 로컬 파일 직접 배치 | `data/embedding_dictionary_e5.json` 두기 | 도커 볼륨 마운트로 그대로 사용 |
| C. 둘 다 미설정 | (아무것도 안 함) | 서버는 기동, 게임 라우트는 503. `/health`로 상태 확인 |

### 사용자가 1줄로 실행 가능해진 케이스
```bash
# 도커 (권장)
git clone <repo>
cd ccomantle
cp .env.example .env  # → EMBEDDING_HF_REPO 값 입력
docker compose up

# 도커 없는 경로
pip install -r requirements.txt
export EMBEDDING_HF_REPO=user/repo
python src/app.py
```

### 추가된 환경변수
| 변수 | 기본값 | 용도 |
|---|---|---|
| `EMBEDDING_HF_REPO` | (없음) | 비어있으면 HF 다운로드 시도 안 함 |
| `EMBEDDING_HF_FILE` | `embedding_dictionary_e5.json` | HF repo 내 파일명 |
| `EMBEDDING_HF_TYPE` | `dataset` | `dataset`/`model`/`space` |
| `EMBEDDING_FILE` | `data/embedding_dictionary_e5.json` | 로컬 경로 오버라이드 |

---

## [6번] QR 시나리오 안정화 — 모바일 + 동시성 + 친화적 UX (2026-05-09)

### 배경
"운영자가 정답 설정 → QR 공유 → 모르는 사람이 모바일로 바로 단어 제출"
시나리오를 차단 없이 만드는 것이 목표. 보안보다 즉시 사용성에 중점.

### 변경 내용

**서버 (동시성 + 회복력)**
- `core/game.py`: `asyncio.Lock` 도입. 모든 mutation 메서드(`reset_for_answer`,
  `set_round`, `submit_guess`, `end`)가 직렬화됨. 여러 팀 동시 제출 시 발생하던
  `team_colors` 덮어쓰기 / `rounds.append` race / 중복검사 race 모두 차단
- `app.py`: 임베딩 파일 없어도 서버 기동. 라우트는 503 반환 (운영자가 컨테이너
  로그 보고 원인 파악 가능, 파일만 따로 올리면 재시작 한 번이면 복구)
- `app.py`: `EMBEDDING_FILE` 환경변수로 경로 오버라이드 가능
- `api/routes.py`: `/health` 엔드포인트 신규. `{ready, words}` 반환
- `api/routes.py`: `/final_result`에 `answer` 필드 추가 (게임 종료 후 정답 공개)
- `api/routes.py`: `get_game` dependency가 임베딩 미로드 시 503으로 실패

**클라이언트 (모바일 + UX)**
- `static/js/game.js`: 한글 IME 조합 중 Enter 무시 (`e.isComposing`, keyCode 229)
- `static/js/game.js`: 단어 거부 시 입력칸 흔들기 + orange 색상 (`showLatest` 헬퍼)
- `static/js/game.js`: 최종 결과 오버레이에 정답 단어 표시
- `static/css/game.css`: 흔들기 애니메이션 + 모바일 `@media (max-width: 768px)`
  분기 (2x2 그리드 → 세로 스택, 폰트 16px+로 iOS 자동줌 방지, 막대 컬럼 숨김)
- `static/css/admin.css`: 모바일 `@media` 분기 (운영자도 폰으로 조작 가능하도록)
- `static/game.html` / `admin.html`: viewport meta 추가
- `static/game.html`: 정답 표시용 `<p id="final-answer">` 추가

**문서**
- `README.md`: 운영자 셋업 3단계 (임베딩 → 로컬 실행 → QR 공유) 가이드
- `docs/CHANGELOG.md`: 본 항목 추가

### 사용자가 친화적 에러 메시지 받게 된 케이스
| 상황 | 변경 전 | 변경 후 |
|---|---|---|
| 사전 미등록 단어 | "사전에 없는 단어입니다." | "사전에 없는 단어입니다. 일반 한국어 명사 약 5만개 중에서 골라주세요." + 입력칸 흔들기 |
| 팀명 누락 | 무반응 | "팀 이름을 먼저 입력해주세요." |
| 네트워크 오류 | 콘솔만 | "네트워크 오류. 잠시 후 다시 시도해주세요." |
| 한글 입력 후 Enter | 조합 중 상태로 제출됨 | IME 조합 완료 후에만 제출 |
| 모바일 접속 | 데스크탑 2x2 그리드 짜부라짐 | 세로 스택, 입력 우선 표시 |

### API 변경 사항
- `/health` 신규
- `/final_result` 응답에 `answer` 필드 추가 (이전 호출부와 호환됨, 추가만)

---

## [5번] Dockerfile 수정 및 requirements 분리 (2026-05-08)

### 문제

**Dockerfile**
- `CMD ["python", "app.py"]` — `app.py`는 `src/` 안에 있어서 컨테이너 시작 시 `No such file or directory` 오류 발생
- `COPY . .` 로 `.git/`, `docs/`, 전처리 스크립트 등 서버에 불필요한 파일까지 모두 이미지에 포함 → 이미지 사이즈 불필요하게 증가
- `.dockerignore` 없음

**requirements.txt**
- `orjson` 누락 → `src/core/embeddings.py`에서 `import orjson` 실패, 서버 시작 시 즉시 `ImportError`
- `sentence-transformers`, `torch`, `jinja2`, `python-multipart` — 서버 런타임에서 실제로 사용하지 않는 패키지. 이미지에서 torch 혼자 수백 MB 차지

### 해결

**Dockerfile 변경**
- `CMD ["python", "app.py"]` → `CMD ["python", "src/app.py"]`
- `COPY . .` → 필요한 디렉터리(`src/`, `static/`, `data/`)만 개별 COPY
- requirements 먼저 COPY·install 후 소스 COPY (레이어 캐시 활용 — 소스만 바뀌면 pip install 스킵)

**requirements.txt** (런타임 전용, 4개)
```
fastapi / uvicorn / numpy / orjson
```

**requirements-dev.txt** (신규, 전처리 전용)
```
torch / transformers / tqdm / scikit-learn
```

**.dockerignore** (신규)
```
.git/, __pycache__/, docs/, requirements-dev.txt, 전처리 스크립트 3개
```

---

## [4번] 정적 파일 CSS/JS 분리 및 CSS 변수화 (2026-05-08)

### 문제
`game.html`(725줄), `admin.html`(357줄) 안에 CSS와 JS가 인라인으로 섞여 있음.
- CSS만 game.html 300줄, admin.html 167줄 → 하나의 파일에서 스타일·로직·구조를 동시에 수정해야 하는 구조
- 두 파일에 동일한 색상값(`#020617`, `#fbbf24` 등)이 하드코딩으로 중복 → 색상 하나 바꾸면 두 파일을 모두 수작업으로 변경해야 함
- 게임 로직 JS 315줄이 HTML 안에 묻혀 있어 가독성과 유지보수성 저하

### 해결

**신규 파일**
```
static/
  css/
    variables.css   # 공유 CSS 커스텀 프로퍼티 (색상 14개, 폰트)
    game.css        # game.html 전용 스타일 (300줄)
    admin.css       # admin.html 전용 스타일 (167줄)
  js/
    game.js         # game.html 전용 스크립트 (폴링·렌더링·추측 제출)
    admin.js        # admin.html 전용 스크립트 (정답설정·라운드·top1000)
```

**변경된 파일**
| 파일 | 변경 전 | 변경 후 |
|---|---|---|
| `static/game.html` | 725줄 (CSS+JS 포함) | 97줄 (HTML 골격 + link/script 태그만) |
| `static/admin.html` | 357줄 (CSS+JS 포함) | 68줄 (HTML 골격 + link/script 태그만) |

**CSS 변수 목록 (variables.css)**
```
--bg, --bg-surface, --border, --border-muted, --border-input,
--text, --text-muted, --accent, --btn-bg, --btn-text,
--green, --orange, --blue, --red
```
game.css·admin.css 양쪽에서 하드코딩 색상을 모두 변수로 교체.
이제 색상 변경은 `variables.css` 한 파일만 수정하면 됨.

### 동작 변경 없음
HTML 구조·CSS 클래스명·JS 함수명 모두 동일 유지.
`/static/css/`, `/static/js/` 경로는 FastAPI의 `StaticFiles` 마운트(`/static`)로 서빙됨.

---

## [3번] 사용하지 않는 전처리 스크립트 정리 (2026-05-08)

### 문제
ver1/ver2 중복 파일 6개가 `src/`에 혼재. 현재 파이프라인에서 사용하지 않거나,
더 나은 버전으로 완전히 대체된 상태.

### 삭제된 파일
| 파일 | 대체 |
|---|---|
| `src/E5_embedding.py` | `E5_embedding_ver2.py` (스케일링 추가) |
| `src/fasttext_extract_50k_words.py` | `make_words_from_vec.py` (더 정교한 필터) |
| `src/fasttext_extract_50k_words_ver2.py` | 임베딩 방식 자체가 FastText → E5로 교체 |
| `src/make_words_from_fasttext.py` | `make_words_from_vec.py` (권장) |
| `src/fasttext_loader.py` | 미사용 (ko-sroberta 로더 잔재) |
| `src/embedding_precompute.py` | `E5_embedding_ver2.py` (ko-sroberta → E5) |

### 현행 전처리 파이프라인 (정리 후)
```
make_words_from_vec.py      → words_50000.json
E5_embedding_ver2.py        → embedding_dictionary_e5_scaled.json
run_embed.py                → 위 두 스크립트 오케스트레이션
```

---

## [2번] 하드코딩 경로 환경변수화 (2026-05-08)

### 문제
전처리 스크립트 5개에 특정 Windows 경로가 하드코딩되어 있어, 다른 머신에서 실행 불가.

**영향 파일 / 변경 전 경로**
| 파일 | 하드코딩 값 |
|---|---|
| `src/fasttext_extract_50k_words.py` | `C:\Users\창영\Desktop\cc.ko.300.bin` |
| `src/fasttext_extract_50k_words_ver2.py` | `C:\Users\창영\Desktop\cc.ko.300.bin` |
| `src/make_words_from_fasttext.py` | `C:\Users\창영\Desktop\fasttext\fasttext.exe` / `.bin` |
| `src/make_words_from_vec.py` | `C:\Users\창영\Desktop\fasttext\cc.ko.300.vec` |
| `src/run_embed.py` | 위 스크립트를 subprocess로 호출 |

### 해결
- 각 스크립트에 `argparse` + `os.getenv` 조합 추가
- 우선순위: CLI 인자 > 환경변수 > 오류 메시지 출력
- `run_embed.py` 는 환경변수를 자식 프로세스에 그대로 전달

### 환경변수 목록
| 변수 | 사용 스크립트 | 설명 |
|---|---|---|
| `FASTTEXT_MODEL_PATH` | extract_50k (both), make_from_fasttext | `.bin` 모델 파일 절대경로 |
| `FASTTEXT_EXE_PATH` | `make_words_from_fasttext.py` | `fasttext.exe` 절대경로 |
| `FASTTEXT_VEC_PATH` | `make_words_from_vec.py` | `.vec` 파일 절대경로 |

---

## [1번] app.py 모듈 분리 + 벡터 행렬화 (2026-05-08)

### 문제
- `src/app.py` 354줄에 전역 상태 11개 + API 로직 + 수학 함수 혼재
- `@app.on_event("startup")` deprecated
- `/top1000` 매 요청마다 5만 번 numpy 변환 + 5만 번 코사인 계산
- HTML 응답 시 매 요청마다 `open().read()` 디스크 IO

### 해결

**신규 파일**
```
src/
  core/
    embeddings.py   # EmbeddingStore: JSON → float32 행렬 (N×D)
    game.py         # GameState 클래스: 모든 상태·로직
  api/
    schemas.py      # Pydantic 요청 모델 3개
    routes.py       # APIRouter + Depends(get_game)
  app.py            # 50줄: lifespan + mount + 3개 HTML 라우트
```

**개선 내용**
| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 상태 관리 | 전역 변수 11개 | `GameState` 단일 인스턴스 (`app.state.game`) |
| 코사인 계산 | 5만 회 루프 | `matrix @ answer_vec` 한 번 (BLAS) |
| 임베딩 저장 | `dict[str, list[float]]` | `np.ndarray (N, D) float32` + 인덱스 dict |
| startup | `@app.on_event` (deprecated) | `@asynccontextmanager lifespan` |
| HTML 서빙 | `open().read()` | `FileResponse` |

**API 호환성**: URL·요청·응답 구조 100% 유지 (프론트엔드 수정 불필요)
