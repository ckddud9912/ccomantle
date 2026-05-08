# 임베딩 시스템

## 개요
한국어 단어 ~5만 개의 벡터를 미리 계산해 JSON으로 저장하고, 서버 시작 시 메모리에 로드한다.
게임 중 코사인 유사도 계산에 사용된다.

## 파일 구조

```
src/core/embeddings.py   — 런타임 임베딩 로드 · 조회
src/E5_embedding.py      — (전처리) E5 임베딩 생성 v1
src/E5_embedding_ver2.py — (전처리) E5 임베딩 생성 v2 (스케일링 추가, 현행)
src/embedding_precompute.py — (전처리) ko-sroberta 임베딩 생성
data/embedding_dictionary_e5.json — 생성된 임베딩 파일 (git 미포함)
```

## EmbeddingStore (core/embeddings.py)

```python
@dataclass
class EmbeddingStore:
    words: List[str]           # 단어 목록, 인덱스와 1:1 대응
    matrix: np.ndarray         # shape (N, D), float32, L2-정규화 완료
    word_to_idx: Dict[str, int]
```

### 설계 결정
- **float32 행렬**: JSON에서 로드 후 즉시 numpy 행렬로 변환. 이후 `matrix @ vec` 한 번으로 전 단어 코사인 동시 계산 가능
- **L2 정규화 재적용**: 임베딩 파일에 이미 정규화가 되어 있지만, 안전하게 로드 시 다시 정규화
- **word_to_idx**: 특정 단어의 벡터를 O(1) 조회

### 유사도 계산
벡터가 정규화되어 있으므로 `a · b == cosine_similarity(a, b)`.
- 단어 vs 단어: `store.vector(w1) @ store.vector(w2)`
- 정답 vs 전체: `store.matrix @ answer_vec` → shape (N,) 한 번에

## 임베딩 생성 파이프라인 (전처리)

```
[FastText .bin/.vec] → 단어 추출 → words_50000.json
                                        ↓
                              E5_embedding_ver2.py
                                        ↓
                              embedding_dictionary_e5.json
```

사용 모델: `intfloat/multilingual-e5-large`
- 입력에 `"query: "` prefix 필수 (E5 스펙)
- CLS 토큰 추출 → L2 정규화
- 스케일링: 1000위 유사도가 0.63이 되도록 전체 벡터를 조정

## 런타임 로드 (app.py lifespan)

```python
store = load_store(DATA_DIR / "embedding_dictionary_e5.json")
app.state.game = GameState(store=store)
```

서버 시작 시 한 번만 로드. 이후 `app.state.game.store`로 접근.
