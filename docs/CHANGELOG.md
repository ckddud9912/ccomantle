# Refactoring Changelog

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
