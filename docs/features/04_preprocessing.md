# 전처리 파이프라인

## 개요
게임 서버가 사용하는 `data/embedding_dictionary_e5.json`을 만들기 위한 1회성 스크립트들. 서버 런타임과 무관.

## 파이프라인 순서

```
1. 단어 추출
   make_words_from_vec.py  (권장)
   또는 fasttext_extract_50k_words_ver2.py
   └→ data/words_50000.json

2. 임베딩 생성
   E5_embedding_ver2.py
   └→ data/embedding_dictionary_e5_raw.json  (중간 저장)
      data/embedding_dictionary_e5_scaled.json  (최종)
```

## 각 스크립트

### `make_words_from_vec.py` (권장)
`.vec` 텍스트 형식 파일에서 한국어 단어 5만 개를 추출한다.

**필터 조건**
- 순수 한국어만 (`[가-힣]+$`)
- 글자 수 2~6자
- 부사 제외 (`게` / `히` 로 끝나는 단어)

**실행**
```bash
# 환경변수
FASTTEXT_VEC_PATH=/path/to/cc.ko.300.vec python src/make_words_from_vec.py

# 또는 CLI 인자
python src/make_words_from_vec.py --vec /path/to/cc.ko.300.vec
```

---

### `fasttext_extract_50k_words_ver2.py`
FastText `.bin` 모델 파이썬 라이브러리로 단어 추출 + PCA 128차원 + L2 정규화.

**실행**
```bash
FASTTEXT_MODEL_PATH=/path/to/cc.ko.300.bin python src/fasttext_extract_50k_words_ver2.py
# 또는
python src/fasttext_extract_50k_words_ver2.py --model /path/to/cc.ko.300.bin
```

---

### `E5_embedding_ver2.py` (현행 임베딩 생성)
`words_50000.json` → `multilingual-e5-large` 모델로 인코딩 → 스케일링 → JSON 저장

**실행**
```bash
python src/E5_embedding_ver2.py
```
GPU 있으면 자동으로 CUDA 사용 (`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`)

**스케일링 (scale_embeddings)**
1. Mean Centering (전체 평균 제거)
2. L2 정규화
3. 스케일 탐색: 1000위 평균 유사도 ≈ 0.63 이 되는 곱셈 계수 찾기 (0.5~3.0 범위, 40 스텝)
4. 최종 L2 정규화 후 저장

---

### `run_embed.py`
오케스트레이터. 파일 존재 여부를 확인하고 없을 때만 각 단계를 실행한다.

```
words_50000.json 없음? → make_words_from_vec.py 실행
embedding_dictionary.json 없음? → embedding_precompute.py 실행
```

**환경변수**
```bash
FASTTEXT_VEC_PATH=/path/to/cc.ko.300.vec python src/run_embed.py
```

---

## 삭제된 파일 (3번 정리)
아래 파일들은 현행 파이프라인에서 불필요하여 제거됨. 이력은 git log 참고.

| 파일 | 이유 |
|---|---|
| `src/E5_embedding.py` | `E5_embedding_ver2.py`(스케일링 추가)로 대체 |
| `src/fasttext_extract_50k_words.py` | `make_words_from_vec.py`로 대체 |
| `src/fasttext_extract_50k_words_ver2.py` | 임베딩 방식이 FastText → E5로 전환됨 |
| `src/make_words_from_fasttext.py` | `make_words_from_vec.py`(권장)로 대체 |
| `src/fasttext_loader.py` | 미사용 (ko-sroberta 로더 잔재) |
| `src/embedding_precompute.py` | `E5_embedding_ver2.py`로 대체 (ko-sroberta 기반) |

## 필요 라이브러리 (전처리 전용)
```
fasttext         # 바이너리 모델 로드
scikit-learn     # PCA
transformers     # E5 모델
torch
tqdm
```
이 패키지들은 서버 운영에 불필요. `requirements-dev.txt` 분리 권장.
