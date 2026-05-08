# API 레퍼런스

## 개요
FastAPI 기반. 모든 엔드포인트는 `src/api/routes.py`에 정의.
`GameState`는 `app.state.game`에서 `Depends(get_game)`으로 주입.

## 엔드포인트

### `POST /set_answer`
정답 단어를 설정하고 게임을 초기화한다.

**요청**
```json
{ "answer": "사과" }
```

**응답**
```json
{ "status": "ok", "answer": "사과" }
// 사전 미등록 시
{ "error": "사전에 없는 단어입니다." }
```

---

### `POST /set_round`
현재 라운드를 변경한다. (1 ~ MAX_ROUNDS)

**요청**
```json
{ "round": 3 }
```

**응답**
```json
{ "status": "ok", "current_round": 3 }
// 범위 초과 시 HTTP 400
{ "error": "Invalid round" }
```

---

### `POST /guess`
팀이 단어를 제출한다.

**요청**
```json
{
  "team": "팀이름",
  "word": "제출단어",
  "team_color": "#ff0000"   // 선택, 최초 제출 시에만 등록
}
```

**응답**
```json
// 정답
{ "result": "correct", "entry": { ... } }

// 일반 제출
{ "result": "ok", "entry": { ... } }

// 이미 이번 라운드에 제출한 팀
{ "result": "duplicate" }

// 오류
{ "error": "..." }
```

**entry 구조**
```json
{
  "round": 1,
  "team": "팀A",
  "team_color": "#ff0000",
  "word": "단어",
  "is_answer": false,
  "rank": 42,
  "similarity": 0.731
}
```

---

### `GET /leaderboard`
현재 게임 전체 상태를 반환한다. 프론트엔드 폴링 대상.

**응답**
```json
{
  "current_round": 2,
  "max_rounds": 6,
  "answer": null,
  "sim_top1": 0.98,
  "sim_top20": 0.85,
  "sim_top1000": 0.63,
  "rounds": {
    "1": [ ...entry 배열, similarity 내림차순... ],
    "2": [],
    ...
  },
  "finished": false
}
```

---

### `GET /top1000`
정답 기준 상위 1000개 단어 목록.

**응답**
```json
{
  "answer": "사과",
  "top1000": [
    { "rank": 1, "word": "사과", "similarity": 1.0 },
    { "rank": 2, "word": "배", "similarity": 0.87 },
    ...
  ]
}
```

---

### `POST /end_game`
경기를 종료한다. 이후 `/guess`는 오류 반환.

**응답**
```json
{ "status": "finished" }
```

---

### `GET /final_result`
팀별 평균 유사도 기준 최종 순위.

**응답**
```json
{
  "result": [
    { "team": "팀A", "team_color": "#ff0000", "avg": 0.821 },
    { "team": "팀B", "team_color": "#0000ff", "avg": 0.654 }
  ]
}
```

---

## HTML 페이지

| URL | 파일 |
|---|---|
| `GET /` | `static/index.html` |
| `GET /game` | `static/game.html` |
| `GET /admin` | `static/admin.html` |

정적 에셋: `GET /static/**` → `static/` 폴더 직서빙
