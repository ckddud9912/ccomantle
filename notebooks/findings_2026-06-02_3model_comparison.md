# [←](../README.md) 임베딩 모델 3-way 비교 — multilingual-e5 vs KoE5 vs KURE (2026-06-02)

> ipynb `embedding_model_comparison.ipynb` 실행 결과. PR #19 의 결정적 발견 ("multilingual-e5-large 가 한국어 의미보다 글자 패턴 cluster") 의 본질 해결 검증.

---

## 핵심 결론

**KoE5 가 ccomantle 게임 컨텍스트에 가장 적합**. KURE 는 의미적으로 가장 정확하지만 단일 정답 게임 결에는 부적합 (동음이의어 분리가 부작용).

→ **KoE5 로 교체 결정**. 단 scaling 적용 (TOP1000=0.63 target) 필수 (raw 는 cosine 0.7-0.9 압축이라 게임 점수 흐림).

**파인튜닝 X**: KoE5 가 사용자 통증 (의미 cluster) 의 대부분 해결. 남은 fail (사과↔자동차) 은 specific edge case — fine-tuning 으로도 보장 안 됨. 시간 대비 효과 작음.

---

## 비교 결과

### 1. 결정적 pair (사과↔배 vs 사과↔자동차)

| pair | e5 | KoE5 | KURE |
|---|---:|---:|---:|
| 사과↔배 | 0.1239 | 0.6450 | **0.4670** |
| 사과↔자동차 | 0.1668 | 0.6609 ⚠️ | **0.4656** ✓ |
| 사과↔포도 | 0.2234 | 0.6678 | 0.4886 |
| 사과↔딸기 | 0.1556 | 0.6778 | 0.4483 |
| 강아지↔고양이 | 0.4946 | 0.7816 | **0.7902** |
| 강아지↔자동차 | 0.2696 | 0.6531 | 0.5711 |

- **KURE 만** 사과↔배 (0.4670) > 사과↔자동차 (0.4656) 정상화
- KoE5 는 여전히 자동차가 더 가까움 (multilingual-e5 base 의 한계 추정)
- 강아지↔고양이 정합성: KURE 가 가장 좋음

### 2. "사과" top 20 — 글자 패턴 vs 의미

| 모델 | top 의 핵심 패턴 |
|---|---|
| **e5** | "사과와"·"사과는" 활용형 + **"-과" 끝 글자 cluster** (제과·인과·여과·안과·내과 등) |
| **KoE5** | 활용형 + 일부 "-과" 잔존 + **"과일"·"과자"·"사과나무"·"사과술" 의미 단어 등장** ★ |
| **KURE** | 활용형 + **"사과 (apologize)" 동음이의어 cluster** (죄송·미안해·미안하다·용서를) ★★ |

### 3. "강아지" top 20 — noise 제거

| 모델 | top 의 noise |
|---|---|
| **e5** | 소시지·마이애미·쭈꾸미·코나미·데미지 (의미 무관 글자 패턴) |
| **KoE5** | 망아지 (운율 비슷) 외엔 거의 정상 (애완동물·반려동물·반려견·고양이) |
| **KURE** | 거의 다 정상 (반려견·반려동물·애완동물·동물·송아지) |

### 4. "사과" 에서 과일들의 rank

| 과일 | e5 rank | KoE5 rank | KURE rank |
|---|---:|---:|---:|
| 배 | 3,888 | **2,590** | 17,708 ⚠️ |
| 포도 | 737 | 874 | 9,732 ⚠️ |
| 바나나 | 125 | **118** | 15,001 ⚠️ |
| 딸기 | 2,316 | **563** | 26,179 ⚠️ |
| 수박 | 992 | **244** | 27,015 ⚠️ |
| 망고 | 1,859 | **472** | 24,150 ⚠️ |
| 레몬 | 1,574 | **453** | 25,016 ⚠️ |
| 복숭아 | 1,270 | **1,099** | 38,766 ⚠️ |

- **KoE5**: 8개 중 6개 rank 큰 폭 개선. 평균 best
- **KURE**: 8개 모두 e5 보다도 나쁨. **과일 cluster 약화**

**원인**: KURE 가 "사과 → 용서·죄송" 쪽으로 cluster → "사과 → 과일" 의미가 약해짐. 동음이의어 분리는 정확하지만 **단일 정답 게임에서는 한쪽 의미만 정답** → 다른 의미 시도 시 점수 매우 낮음.

---

## 모델별 종합

| 측면 | e5 | KoE5 | KURE |
|---|---|---|---|
| 글자 패턴 cluster | ⚠️ 심함 | △ 일부 잔존 | ✓ 사라짐 |
| 의미 cluster (일반) | △ 들쭉날쭉 | ✓ 개선 | ✓✓ 가장 정확 |
| 과일 cross-check | bad | **best** | worst |
| 사과↔자동차 fail | ⚠️ | ⚠️ | ✓ 풀림 |
| noise 단어 (마이애미·코나미 등) | 다수 | 거의 없음 | 거의 없음 |
| 동음이의어 처리 | 한 의미 | 한 의미 | 다중 의미 (게임엔 ⚠️) |
| **ccomantle 게임 적합도** | △ | **✓✓** | △ |

---

## 결정 — KoE5 로 교체

### 왜 KoE5 인가
- 과일 cross-check rank 평균 best (배·딸기·수박·망고·레몬 대폭 개선)
- 의미 단어 등장 ("과일"·"과자"·"사과나무"·"바다"·"주택")
- e5 의 random noise (마이애미·쭈꾸미·집총) 사라짐
- 단일 정답 게임에 맞는 결 (동음이의어 cluster 분산 X)

### KURE 보류 이유
- 의미적으로는 가장 정확하지만 ccomantle 게임 단일 정답 컨텍스트엔 부적합
- 동음이의어 분리 부작용: 과일 cross-check 가 e5 보다도 나쁨
- 미래에 동음이의어 명확히 처리하는 게임 (예: 정답에 의미 태그) 으로 확장하면 KURE 가 best — **장기 후보로 handoff 에 박아둠**

### KoE5 의 한계 인정
- "사과↔자동차" specific edge case 여전 (multilingual-e5 base 한계)
- 활용형 (조사) 이 여전히 top — 별 트랙 (`feat/dict-cleanup-activations`) 으로 다룰 영역

### 파인튜닝 결정
**X**. KoE5 가 80% 해결. 남은 5-10% 추가 위해 며칠~주 fine-tuning 은 가성비 X. 대신:
- `feat/sim-calibration` — 점수 보정 강화로 남은 fail 영향 줄임
- `feat/game-log-rejected` — 실측 사용자 통증 데이터 수집

---

## 다음 단계 (이번 PR)

1. KoE5 임베딩에 scaling 적용 (TOP1000=0.63 target)
2. `data/embedding_dictionary_e5.json` 백업 + KoE5 scaled 로 swap
3. 게임 테스트 — 정답 "사과" 에서 과일들 점수 자연스러운지 확인
4. HF Hub 재업로드
5. PR commit + push

---

## 관련 파일

- [`embedding_model_comparison.ipynb`](embedding_model_comparison.ipynb) — 본 비교의 원본 ipynb
- [`findings_2026-06-02.md`](findings_2026-06-02.md) — PR #19 의 첫 발견
- [`tools/embedding_eval/build_alt_embeddings.py`](../tools/embedding_eval/build_alt_embeddings.py) — 모델별 임베딩 생성 도구
- 다음 (이번 PR 작성 예정): `tools/embedding_eval/apply_scaling_and_swap.py` — scaling + swap 도구
