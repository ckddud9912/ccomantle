# 게임 로직

## 개요
팀 대항 유사도 추측 게임. 운영자가 정답 단어를 설정하면 팀들이 라운드별로 단어를 제출하고, 정답과의 코사인 유사도를 경쟁한다.

## 파일
```
src/core/game.py
```

## GameState 클래스

모든 게임 상태를 하나의 인스턴스로 관리. `app.state.game`에 저장됨.

### 주요 상태
| 필드 | 타입 | 설명 |
|---|---|---|
| `store` | `EmbeddingStore` | 임베딩 참조 (변경 없음) |
| `answer_word` | `str \| None` | 현재 정답 단어 |
| `answer_vector` | `np.ndarray \| None` | 정답 벡터 (float32) |
| `word_to_rank` | `dict[str, int]` | 정답 기준 전체 단어 순위 |
| `sim_alpha` | `float` | 유사도 스케일링 지수 |
| `current_round` | `int` | 현재 라운드 (1~MAX_ROUNDS) |
| `rounds` | `dict[int, list[dict]]` | 라운드별 제출 내역 |
| `team_colors` | `dict[str, str]` | 팀명 → 색상 (최초 제출 시 고정) |
| `finished` | `bool` | 게임 종료 여부 |

### 상수
```python
MAX_ROUNDS = 6
TARGET_TOP1000 = 0.63  # 1000위 유사도 목표값 (스케일링 기준)
```

## 핵심 로직

### 정답 설정 (`reset_for_answer`)
1. 임베딩 사전에 단어가 있는지 확인
2. 게임 상태 초기화 (rounds, team_colors, finished, current_round)
3. `_compute_rankings()` 호출

### 랭킹 계산 (`_compute_rankings`)
```python
sims = store.matrix @ answer_vector   # (N,) 전체 유사도 한 번에
order = np.argsort(-sims)             # 내림차순 정렬 인덱스
word_to_rank = {word: rank+1 ...}
```
SIM_ALPHA 계산: `(sim_1000th_raw) ** alpha = 0.63` 을 만족하는 alpha
```
alpha = log(0.63) / log(sim_1000th_raw)
```
1000위 유사도가 0 이하이면 alpha = 1.0 (스케일링 없음)

### 추측 처리 (`submit_guess`) 처리 순서
1. 경기 종료 여부 확인
2. 정답 설정 여부 확인
3. 팀 색상 최초 등록 (이후 변경 불가)
4. 해당 라운드에 같은 팀 중복 제출 여부 확인 → `"duplicate"`
5. 정답 일치 → `"correct"`, similarity = 1.0
6. 사전 미등록 단어 → error
7. 유사도 계산 → `"ok"` + entry 반환

### 유사도 스케일링
```python
scaled = max(raw, 0) ** sim_alpha
```
음수 유사도는 0으로 클램프. 양수는 거듭제곱으로 분포를 TARGET_TOP1000 기준으로 조정.

## 게임 흐름
```
set_answer() → [set_round() → guess() × N팀] × MAX_ROUNDS → end_game() → final_result()
```
`leaderboard()`는 언제든지 폴링 가능.
