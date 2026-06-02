"""
tools/embedding_eval/build_alt_embeddings.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
무엇 (What)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
현 ccomantle 사전 (60k 단어) 의 단어 리스트에 대해, 임의의 HuggingFace 임베딩
모델로 새 임베딩 생성. 모델 비교 용도 (multilingual-e5-large 의 한국어 처리
한계 검증 → KoE5/KURE 등 한국어 특화 모델과 spot-check 비교).

왜 (Why)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PR #19 (notebooks/findings_2026-06-02.md) 의 발견: multilingual-e5-large 가
한국어 의미보다 글자 패턴에 cluster 됨. 한국어 특화 모델 비교가 본질 해결
검증의 핵심. 본 스크립트가 비교 입력 생성.

차이점 (vs src/E5_embedding_ver2.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 모델·prefix·출력 경로 옵션화
- scaling 단계 X (raw 임베딩, L2 정규화만) — 비교 목적이라 raw 가 정직
- 단어 리스트는 기존 사전 (`data/embedding_dictionary_e5.json`) 의 키 사용

사용 (Usage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pip install -r requirements-dev.txt  # torch + transformers + tqdm

    # KoE5 (multilingual-e5 한국어 finetune, 같은 prefix)
    python tools/embedding_eval/build_alt_embeddings.py \\
      --model nlpai-lab/KoE5 \\
      --output data/embedding_dictionary_koe5.json

    # KURE-v1 (BGE-M3 한국어, prefix 없음)
    python tools/embedding_eval/build_alt_embeddings.py \\
      --model nlpai-lab/KURE-v1 \\
      --output data/embedding_dictionary_kure.json \\
      --prefix ""
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_batch(
    tokenizer,
    model,
    texts: list[str],
    device: str,
    prefix: str,
    max_length: int = 32,
) -> np.ndarray:
    inputs = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
    enc = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**enc)
        # CLS pooling (E5 / KoE5 / 대부분 sentence encoder 의 standard)
        cls = out.last_hidden_state[:, 0, :]
    cls = cls / cls.norm(dim=1, keepdim=True)
    return cls.cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description="대안 임베딩 모델로 ccomantle 사전 임베딩 재생성")
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace 모델 이름 (예: nlpai-lab/KoE5, nlpai-lab/KURE-v1)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="출력 JSON 경로 (예: data/embedding_dictionary_koe5.json)",
    )
    parser.add_argument(
        "--prefix",
        default="query: ",
        help="입력 텍스트 prefix (default 'query: ' = E5 표준). 빈 문자열은 ''로 명시",
    )
    parser.add_argument(
        "--source-dict",
        type=Path,
        default=DATA_DIR / "embedding_dictionary_e5.json",
        help="단어 리스트 source — 이 사전의 키를 사용",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=32)
    args = parser.parse_args()

    print("=" * 60)
    print(f"대안 임베딩 생성 — {args.model}")
    print("=" * 60)

    # 1. 단어 리스트
    src_path = args.source_dict
    if not src_path.is_absolute():
        src_path = REPO_ROOT / src_path
    if not src_path.exists():
        print(f"❌ source 사전 없음: {src_path}")
        return 1
    print(f"\n[1/3] 단어 리스트 로드: {src_path.relative_to(REPO_ROOT)}")
    with open(src_path, "rb") as f:
        words = list(json.loads(f.read()).keys())
    print(f"   단어 수: {len(words):,}")

    # 2. 모델 로드
    device = _device()
    print(f"\n[2/3] 모델 로드 — device={device}")
    print(f"   모델: {args.model}")
    print(f"   prefix: {args.prefix!r}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()
    print(f"   로드 완료")

    # 3. 임베딩 생성
    print(f"\n[3/3] 임베딩 생성 (batch_size={args.batch_size}, max_length={args.max_length})")
    emb_dict: dict[str, list[float]] = {}
    for i in tqdm(range(0, len(words), args.batch_size)):
        batch = words[i : i + args.batch_size]
        vectors = encode_batch(tokenizer, model, batch, device, args.prefix, args.max_length)
        for w, vec in zip(batch, vectors):
            emb_dict[w] = vec.tolist()

    # 저장
    out_path = args.output
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n저장 중: {out_path.relative_to(REPO_ROOT)}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(emb_dict, f, ensure_ascii=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"   완료: {len(emb_dict):,} 단어, {size_mb:.1f} MB")

    print("\n다음 단계:")
    print(f"  → notebooks/embedding_model_comparison.ipynb 에서 이 사전 로드해 비교")
    return 0


if __name__ == "__main__":
    sys.exit(main())
