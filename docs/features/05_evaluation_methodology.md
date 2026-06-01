# [←](../../README.md) 평가 방법론

> **목적**: ccomantle 임베딩 사전 + 유사도 점수의 품질을 정량 평가하고 개선하기 위한 방법론·진단 절차·측정 지표를 PR 단위로 누적 기록.
>
> **본 게임의 평가 특수성**: ccomantle 은 RAG 응답 평가가 아니라 **(a) 임베딩 사전 자체의 품질** + **(b) 유사도 검색 결과의 의미 정합성** 평가가 본질. RAGAS·prompt engineering 같은 RAG 결 도구는 본 프로젝트 결이 아님 — handoff.md "Anna 정렬 평가 트랙" 참조.
>
> **상위 문서**: `~/GitStudy/make_portfolio/지원/anna/평가전략_종합.md` §4 (ccomantle 적용 전략).

---

## 0. 현재 평가 / 점수 계산 흐름 (코드 기반)

본 게임의 "점수" 가 어떻게 산출되는지 코드 위치 명시.

### 0.1 단어 사전

| 단계 | 코드 | 동작 |
|---|---|---|
| 단어 추출 | [`src/make_words_from_vec.py`](../../src/make_words_from_vec.py) | ko-FastText `.vec` 에서 빈도 순으로 50,000 단어 추출 |
| 필터 | `is_valid_word` | (1) `^[가-힣]+$` 한글만, (2) 길이 2-6, (3) "-게"/"-히" 끝 부사 제외 |
| 임베딩 생성 | [`src/E5_embedding_ver2.py`](../../src/E5_embedding_ver2.py) | `intfloat/multilingual-e5-large` 로 5만 단어 임베딩 → `data/embedding_dictionary_e5.json` |
| 로드 | [`src/core/embeddings.py`](../../src/core/embeddings.py) | L2 정규화, float32 matrix |

### 0.2 유사도 → 게임 점수

[`src/core/game.py`](../../src/core/game.py) 의 `_compute_rankings` (정답 설정 시 1회):

```
# 1. 정답 벡터와 모든 단어의 cosine 계산 → 전체 순위(rank) 산출
sims = store.matrix @ answer_vector          # (50000,)
order = np.argsort(-sims)
word_to_rank = {word: rank+1 for rank, idx in enumerate(order)}

# 2. 1000위 유사도가 TARGET_TOP1000 (=0.63) 이 되도록 거듭제곱 α 보정
#    raw 점수 → x^α (양수만)
sim_alpha = log(0.63) / log(sims[1000위])
sim_top1   = top1   ^ α
sim_top20  = top20  ^ α
sim_top1000= top1000^ α   # ≈ 0.63 으로 맞춰짐
```

**의미**:
- raw cosine 그대로 쓰면 정답 1위 점수 ~0.99, 20위 ~0.85, 1000위 ~0.7 같은 압축 분포 → 사용자가 진행감 느끼기 어려움
- `α` 보정으로 1000위가 항상 0.63 이 되도록 **모든 점수를 늘림** (또는 압축). 정답에 가까울수록 점수 차이가 커 보이게 됨

### 0.3 게임 응답 형태

[`src/core/game.py`](../../src/core/game.py) `submit_guess`:
- 정답 = `is_answer=True, similarity=1.0`
- 사전에 있음 → `rank` (전체 50,000 중) + `similarity` (α 보정값)
- 사전에 없음 → `"사전에 없는 단어입니다."` 에러

→ **사용자가 경험하는 "점수의 품질" 은**:
1. 같은 정답에서, 의미 비슷한 단어가 정말 가까운 순위로 나오는가
2. 의미 다른 단어가 정말 먼 순위로 나오는가
3. **사용자가 떠올린 단어가 사전에 존재하는가** (← 본 PR 직격)

---

## 1. 누락 단어 탐지 (Dictionary Coverage) ★

**현 PR (`feat/dict-quality-report`) 의 1순위 작업**.

### 1.1 문제 정의

게임 플레이 중 사용자가 떠올린 일반 한국어 단어가 **"사전에 없습니다"** 로 거부되는 경우가 잦음. 원인 가설:

| 가설 | 설명 | 검증 방법 |
|---|---|---|
| **(H1) 빈도 cap (50k)** | FastText `.vec` 의 상위 50k 토큰 안에 못 든 일반 명사가 빠짐 | 외부 reference 사전과 set 차이 |
| **(H2) 활용형이 50k 자리 점유** | "있다"/"있는"/"있습니다" 같은 활용형 3-5개가 한 자리씩 차지해 다른 명사를 밀어냄 | 50k 안의 활용형 비율 정량 |
| **(H3) 필터 과잉** | 길이 2-6 / "-게/-히" 부사 제외 / 한글-only 제약이 합법 단어 거부 | 거부된 토큰 샘플 검토 |
| **(H4) 외래어/고유명사 누락** | 한국어로 정착한 외래어 ("커피", "버스"), 고유명사 빠짐 | 위와 동일 |

H1+H2 가 가장 가능성 큼.

### 1.2 진단 접근 — 두 갈래

**(a) 외부 reference 사전과 비교** (이번 PR 의 중심)
- ccomantle 사전(5만) ∩ reference 명사 set → 누락 단어 식별
- 산출: `data/missing_words_candidates.json` (reference 에는 있는데 ccomantle 에 없는 단어 + 그 빈도/우선순위)

**(b) 게임 플레이 로그 분석** (다음 PR 후보)
- 사용자가 시도했다가 거부된 단어의 빈도 집계
- 실제 통증 직접 반영. 단 게임 로그 수집 메커니즘 신규 구축 필요 (현재 X)
- → 본 PR 에선 (a) 만, (b) 는 `feat/game-log-rejected-words` 로 분리

### 1.3 Reference dict 후보 비교

| 후보 | 출처 / 라이선스 | 크기 | 장점 | 단점 |
|---|---|---|---|---|
| **mecab-ko-dic NNG.csv (raw)** ★ | [bitbucket.org/eunjeon/mecab-ko-dic](https://bitbucket.org/eunjeon/mecab-ko-dic/) / Apache 2.0 | NNG ~800k entry | POS별로 미리 분리된 CSV — `NNG.csv` 만 갖다 쓰면 끝. **시스템 mecab 설치 불요**. 한국어 NLP 사실상 표준 사전 → coverage 해석 직관적 | 행 수 많아 메모리 ~50MB |
| **NIADic** | [haven-jeon/NIADic](https://github.com/haven-jeon/NIADic) / MIT | ~250k | 구조화된 한국어 사전, POS 포함, pip 설치 간단 | mecab 보다 작음. mecab 과 어휘 중복 큼 |
| **KoNLPy + mecab 시스템 설치** | KoNLPy / GPL | mecab-ko-dic 동일 | 향후 형태소 분석에 그대로 활용 가능 | macOS/Linux 시스템 빌드 깨지기 쉬움 (brew + JPype + Java) → 첫 PR 의존성 부담 큼 |
| 국립국어원 우리말샘 | [opendict.korean.go.kr](https://opendict.korean.go.kr) / CC BY-SA 2.0 KR | 1M+ | 공식 개방형 사전 | API/덤프 처리 복잡, 표제어 다수 |
| 한국어 위키 명사 | [ko.wikipedia.org dumps](https://dumps.wikimedia.org/kowiki/) / CC BY-SA | 가변 | 동시대 어휘·고유명사 | 비단어 다수, 추출 처리 필요 |
| 빈도 reference (Modu/Sejong corpus) | 국립국어원 / 신청 필요 | corpus 의존 | 진짜 사용 빈도 | 접근 절차 |

#### 1.3.1 결정 — 이번 PR: **mecab-ko-dic NNG.csv raw 단독**

**3가지 이유**:

**(1) 의존성 0**

mecab-ko-dic raw CSV 는 단순 텍스트 파일. 다운로드 후 파싱만 하면 끝.
- 비교: KoNLPy + mecab 시스템 설치는 macOS/Linux 에서 깨지기 쉬움 (brew install, source build, JPype Java binding, Python wheel 호환성)
- 한국어 NLP 프로젝트의 첫 PR 가 mecab 설치에서 막히는 일이 흔함 → 회피
- ccomantle 의 docker-compose 에 mecab 시스템 의존성 추가하면 이미지 빌드 복잡도 ↑

**(2) POS 별 분리 + 표준성 무료**

mecab-ko-dic 는 **이미 POS 별 CSV 로 분리**돼 있음:
```
NNG.csv  - 일반명사 (general noun)     ← 우리가 쓸 것
NNP.csv  - 고유명사 (proper noun)
NNB.csv  - 의존명사
VV.csv   - 동사
VA.csv   - 형용사
...
```
- 우리에게 필요한 게 "한국어 일반 명사 목록" → `NNG.csv` 그대로 갖다 쓰면 됨
- 동사·부사·조사·어미·활용형 안 섞임 (필터 코드 불요)
- mecab-ko-dic 는 한국어 NLP 사실상 표준 → coverage 점수가 학계·업계에서 해석 가능

**(3) 향후 확장 가능 (레이어 추가식)**

- 지금: raw CSV 만으로 충분
- 나중 §2 활용형 클러스터링 단계에 진짜 형태소 분석이 필요해지면 그때 KoNLPy + mecab 추가
- NIADic 도 coverage 결과 의심스러우면 cross-check 로 추가

#### 1.3.2 `NNG.csv` 의 행 구조 (참고)

```
사과,1781,3559,3056,NNG,*,F,사과,*,*,*,*
컴퓨터,1781,3559,3122,NNG,*,F,컴퓨터,*,*,*,*
```

스키마:
| 컬럼 | 의미 |
|---|---|
| 0 | 표제어 ← **우리가 쓸 것** |
| 1 | left_context_id |
| 2 | right_context_id |
| 3 | cost (낮을수록 빈도 ↑) ← **우선순위 산정에 사용 가능** |
| 4 | POS (NNG) |
| 5-11 | 의미·발음·복합어 정보 |

→ 추출 코드는 단순히 `cols[0]` (표제어) + `cols[3]` (cost) 두 개만 보면 됨.

#### 1.3.3 KoNLPy 와의 관계 (왜 KoNLPy 안 쓰는지)

- `KoNLPy.Mecab().nouns(text)` 는 **주어진 텍스트의 명사 추출** — 우리가 원하는 "전체 명사 사전" 이 아님
- 사전 자체를 얻으려면 결국 raw CSV 봐야 함
- 그러면 mecab 시스템 설치 안 하고 CSV 만 다운로드하는 게 합리적
- KoNLPy 는 §2 (활용형 클러스터링) 같은 분석 작업에서 다시 검토

### 1.4 측정 지표

본 PR 의 산출 지표:

| 지표 | 정의 | 의미 |
|---|---|---|
| **Coverage** | `\|ccomantle 사전 ∩ reference NNG\| / \|reference NNG\|` | reference 명사 중 몇 % 가 ccomantle 에 있는가 |
| **Missing count** | `\|reference NNG \ ccomantle 사전\|` | 절대 누락 수 |
| **Missing top-K by frequency** | 누락 단어 중 reference 빈도(또는 ko-FastText 빈도) 상위 K | 추가 우선순위 |
| **Filter rejection breakdown** | 누락 단어 중 (a) 길이 제약 (b) 한글 외 (c) 부사 (d) cap 50k 초과 비율 | 어느 필터가 가장 많이 누락시키는지 |

### 1.5 우선순위 산정

추가할 단어 후보는 다음 점수로 정렬:

```
priority_score = reference_frequency * 1.0
               + already_in_fasttext_50k_to_100k * 0.5      # 임베딩 즉시 가능
               - is_proper_noun_only_in_kowiki * 0.3        # 고유명사는 게임에 부적합
```

상위 N (예: 1000) 을 첫 추가 batch 후보로.

### 1.6 산출

본 PR 의 산출물:

| 파일 | 내용 |
|---|---|
| `data/missing_words_candidates.json` | reference 에 있고 ccomantle 에 없는 단어 + priority_score + 누락 원인 (cap/필터/완전 부재) |
| `data/quality_report.json` | coverage·missing count·필터별 누락 비율 등 요약 통계 |
| `tools/dict_quality/coverage_check.py` (신규) | reference 사전 로드 + 비교 + 위 두 JSON 생성 스크립트 |
| 본 doc §1.7 | 실측 결과 (PR 완료 시 채워짐) |

### 1.7 결과 (2026-06-01 실측)

[`tools/dict_quality/coverage_check.py`](../../tools/dict_quality/coverage_check.py) 실행 결과 (mecab-ko-dic 2.1.1-20180720 기준).

#### 핵심 수치

| 항목 | 값 |
|---|---|
| ccomantle 사전 | 50,000 |
| mecab NNG (일반명사 표제어) | 205,269 |
| 교집합 | 9,876 |
| **Coverage (NNG ∩ ccm / NNG)** | **4.81%** |
| 누락 단어 | 195,393 |

#### 누락 단어 필터 거부 원인 분류

| 원인 | 개수 | 비율 |
|---|---|---|
| `passes_filter` (ccomantle 필터 통과 — cap 50k 초과 또는 FastText 부재) | 192,260 | **98.4%** |
| `too_long` (>6 자) | 1,800 | 0.9% |
| `too_short` (<2 자) | 990 | 0.5% |
| `adverb_pattern` ("게/히" 끝) | 314 | 0.2% |
| `not_pure_korean` | 29 | 0.0% |

→ 압도적 다수가 **빈도 cap 또는 FastText 부재** 원인. 필터 자체로 거부되는 건 약 1.6% 뿐.

#### Top 5000 누락 단어의 cost 분포 (mecab cost ↓ = 빈도 ↑)

| cost 범위 | 개수 |
|---|---|
| `<0` (매우 빈도 높음) | 2 |
| `0~999` | 89 |
| `1000~4999` | 4,909 |
| `5000+` | 0 |

#### 결정적 발견 — 필터 자체 결함

**finding 1: 1글자 명사 전면 누락** (too_short 의 본질)

게임에서 매우 흔한 단어가 막혀있음:
```
끝(67), 말(1127), 글(1172), 땅(1386), 돈(1419), 꾼(1429),
때(1513), 홈(1611), 탓(1615), 꿈(1621), 봄(1649), 날(1700),
칸(1745), 쌀(1807), 값(1808), 눈(1812), 밤(1865), 벗(1871) ...
```
- 현 코드 [make_words_from_vec.py:18](../../src/make_words_from_vec.py#L18): `if not (2 <= len(word) <= 6)` → 1글자 막음
- 1글자 한국어 명사는 매우 흔하고 게임에 필수. **첫 번째 cleanup 대상**

**finding 2: 부사 필터가 명사도 같이 잡음**

```
가게(260), 무게(1525)  ← reason=adverb_pattern
```
- 현 코드 [make_words_from_vec.py:12](../../src/make_words_from_vec.py#L12): `ADVERB_PATTERN = r".+(게|히)$"`
- 진짜 부사 ("빠르게", "조용히") 거부하려다 **명사 "가게"/"무게"/"모기" 등도 같이 거부**
- 정밀한 해결: 단순 패턴 매칭 X, mecab NNG 명사 사전 참조해서 "명사면 통과" 시그널 추가
- 차선책: 부사 필터 자체를 제거 (FastText 빈도 cap 이 어차피 진짜 부사 다수 거름)

**finding 3: 진짜 흔한 명사 2개가 cap 또는 FastText 부재**

```
학년도(-335), 큰일(-12)  ← passes_filter, 즉 필터 통과인데 사전엔 없음
```
- "학년도" 는 매우 흔한 한국어 단어인데 ko-FastText `.vec` 의 top 50,000 안에 없거나, FastText 가 "학년"+"도" 로 분리해서 처리한 듯
- → **빈도 cap 확장만으로 해결 불가** 가능성. 외부 사전 단어를 직접 보강해야 할 수도

**finding 4: too_long 의 다수는 게임 부적합**

```
메이예르홀트주의, 범아프리카주의, 유니테리언주의, 아리스토텔레스주의 ...
```
- 6글자 cap 자체는 게임 UX 기준 합리적 (긴 학술 용어는 게임에 부적합)
- 단 일반 명사 중 7-8글자가 있는지 추가 검토는 필요

### 1.8 다음 단계 — 우선순위 갱신 (본 결과 반영)

§1.7 결과 보면 두 트랙으로 나누는 게 자연스러움:

#### 트랙 A — 필터 결함 수정 (작은 PR, 즉시)

**PR `feat/dict-filter-fix`** (1-2일):
1. `make_words_from_vec.py` 의 `is_valid_word` 수정:
   - 길이 최소 2 → 1 로 완화 (1글자 명사 허용)
   - `ADVERB_PATTERN` 제거 또는 mecab NNG 참조로 정밀화
2. `data/words_50000.json` 재생성 (FastText 가 PC 에 있어야 함 — 환경 의존)
3. 또는 즉시 패치: 현 50,000 단어에 finding 1·2 의 누락 흔한 명사 한 번에 보강 (벡터는 e5 로 재생성)
4. coverage_check.py 재실행해 coverage 변동 확인

이 트랙은 **finding 1·2 (필터 결함, 약 1,300 단어)** 만 다룸. cap/FastText 문제와는 독립.

#### 트랙 B — 어휘 보강 (중간 PR)

**PR `feat/dict-expand-coverage`** (1주):
1. `missing_words_candidates.json` 상위 후보 검토 (수동 spot-check — "주의"·"제도"·"파" 류 학술/조어 제외)
2. ko-FastText `.vec` 에 벡터가 있는지 확인 → 있으면 즉시 추가
3. 없는 단어 → multilingual-e5-large 로 추가 임베딩 생성
4. 50,000 cap 폐지 또는 확장 (예: 100,000)
5. `embedding_dictionary_e5.json` 재생성
6. game.py 의 `_compute_rankings` 1000위=0.63 보정값 변동 확인

#### 트랙 C — 진단 보강 (별 PR, 선택)

**PR `feat/dict-game-log-rejected`**:
- 게임 클라이언트에 거부 단어 collection 추가 (사용자 동의 가정 X — 단순 서버 로그 기록)
- 실측 데이터로 §1.7 의 mecab 기반 가설 검증
- "사용자가 실제로 시도한 누락 단어" vs "mecab 기반 추정 누락" 일치 비율

#### 진행 권장 순서

1. **본 PR (현재) `feat/dict-quality-report` 머지**: 진단 인프라 + 결과 박제
2. **다음 PR `feat/dict-filter-fix`**: finding 1·2 직접 해소 (가장 작은 PR, 가장 큰 사용자 체감 효과 추정)
3. **그다음 `feat/dict-expand-coverage`**: 어휘 보강 (트랙 B)
4. **여유 시 트랙 C**: 게임 로그 수집

---

## 2. 활용형 / 노이즈 / 외래어 검사 (다음 PR — `feat/dict-cleanup-noise`)

_(placeholder — 본 PR 의 §1.6 산출이 누락 단어 추가의 baseline 이라면, 본 §은 그 baseline 이 깨끗한지 검증)_

**계획**:
- 활용형 클러스터링: 같은 어간의 활용형 그룹 ("있다"/"있는"/"있었다") → 대표형만 남기는 결정
- 외래어 정상화: "커피"·"버스" 같이 정착된 외래어는 유지, "TPO"·"API" 같은 약어는 제외
- 깨진 단어: 의미 없는 모음·자음 결합 검출
- 측정: NIA v3.5 비정형 8지표 중 "노이즈"·"중복도"·"언어품질" 매핑

---

## 3. 임베딩 의미 정합성 — intrinsic 평가 (다음 PR — `feat/embedding-intrinsic-eval`)

_(placeholder)_

**계획**:
- 인간 직관 유사도 vs 임베딩 cosine 의 Spearman 상관
- 골든셋: KorLex Golden Similarity (2866 pair) 또는 WordSim-353 한국어 + KATS Analogy
- 본 게임 사전에 한정해 (5만 안의 pair 만) 점수
- 측정: NIA v3.5 비정형 8지표 중 "내용품질" 매핑

---

## 4. 점수 보정 (sim_alpha) 검증 (다음 PR — `feat/sim-calibration`)

_(placeholder)_

**계획**:
- 현 [`game.py:78-80`](../../src/core/game.py#L78-L80) 의 `1000위 = 0.63` target 이 왜 0.63 인지 근거 없음
- 다양한 정답 단어에 대해 α 값 분포 측정 → 안정적 보정인지 검증
- 사용자 plays 데이터에서 "점수 X 에서 사용자가 멈추는 비율" 등 calibration ↔ UX 연결 분석
- 측정: NIA v3.5 정형 6지표 중 "신뢰성" 매핑

---

## 5. 게임 로그 기반 검증 (장기 — `feat/game-log-rejected-words` + `feat/game-log-eval`)

_(placeholder)_

**계획**:
- "사전에 없습니다" 거부된 단어 집계 (사용자가 실제 시도한 단어)
- 정답 추측 분포 vs 임베딩 유사도 분포 상관
- 측정: NIA v3.5 비정형 8지표 중 "내용품질"·"다양성" 실사용 검증

---

## 6. 변경 이력

- **2026-06-01 v1**: 골격 + §0 현재 평가 흐름 + §1 누락 단어 탐지 방법론. `feat/dict-quality-report` 의 첫 작업. §2-5 는 placeholder.
