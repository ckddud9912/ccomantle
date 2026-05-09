# Refactoring Changelog

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
