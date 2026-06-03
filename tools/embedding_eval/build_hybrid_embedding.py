"""
tools/embedding_eval/build_hybrid_embedding.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
무엇 (What)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
두 임베딩 사전 (예: KoE5 + KURE) 의 같은 단어 벡터를 결합해 hybrid 임베딩
생성. PR #21 에서 KoE5 가 게임 적합·KURE 가 의미 정확 — 둘의 장점을 결합
시도.

방법
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- concat: 두 vector concatenate (1024 + 1024 = 2048 dim). 정보 보존 max
- average: weighted average (1024 dim). 차원 유지, 계산 빠름

각 입력 vector 는 미리 L2 정규화, 결합 후 다시 L2 정규화.

사용
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # KoE5 + KURE concat (2048 dim)
    python tools/embedding_eval/build_hybrid_embedding.py \\
      --source-a data/embedding_dictionary_koe5.json \\
      --source-b data/embedding_dictionary_kure.json \\
      --output data/embedding_dictionary_hybrid_koe5_kure_concat.json \\
      --mode concat

    # KoE5 + KURE average (1024 dim, KoE5 가중치 0.7)
    python tools/embedding_eval/build_hybrid_embedding.py \\
      --source-a data/embedding_dictionary_koe5.json \\
      --source-b data/embedding_dictionary_kure.json \\
      --output data/embedding_dictionary_hybrid_koe5_kure_avg.json \\
      --mode average --weight-a 0.7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, list[float]]:
    with open(path, "rb") as f:
        return json.loads(f.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="두 임베딩 사전 결합 → hybrid 임베딩")
    parser.add_argument("--source-a", required=True, type=Path)
    parser.add_argument("--source-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["concat", "average"], default="concat")
    parser.add_argument(
        "--weight-a",
        type=float,
        default=0.5,
        help="average 모드의 source-a 가중치 (0-1, default 0.5)",
    )
    args = parser.parse_args()

    sa = args.source_a if args.source_a.is_absolute() else REPO_ROOT / args.source_a
    sb = args.source_b if args.source_b.is_absolute() else REPO_ROOT / args.source_b
    out = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    for p in (sa, sb):
        if not p.exists():
            print(f"❌ source 없음: {p}")
            return 1

    print("=" * 60)
    print(f"hybrid 임베딩 생성 ({args.mode})")
    print("=" * 60)

    print(f"\n[1/3] 로드")
    a = _load(sa)
    b = _load(sb)
    print(f"   a: {sa.name} → {len(a):,} 단어")
    print(f"   b: {sb.name} → {len(b):,} 단어")

    common = sorted(set(a.keys()) & set(b.keys()))
    print(f"   교집합 (둘 다 있는 단어): {len(common):,}")
    only_a = len(a) - len(common)
    only_b = len(b) - len(common)
    if only_a or only_b:
        print(f"   (a-only: {only_a:,}, b-only: {only_b:,} — hybrid 에서 제외됨)")

    print(f"\n[2/3] hybrid 생성 (mode={args.mode}"
          + (f", weight_a={args.weight_a}" if args.mode == "average" else "")
          + ")")
    hybrid: dict[str, list[float]] = {}
    wa = args.weight_a
    wb = 1.0 - args.weight_a
    for w in common:
        va = np.array(a[w], dtype=np.float32)
        vb = np.array(b[w], dtype=np.float32)
        # 각각 L2 정규화 (이미 됐을 수도 있지만 안전)
        va = va / max(np.linalg.norm(va), 1e-8)
        vb = vb / max(np.linalg.norm(vb), 1e-8)

        if args.mode == "concat":
            vh = np.concatenate([va, vb])
        else:  # average
            vh = wa * va + wb * vb

        # 결합 후 다시 L2 정규화
        vh = vh / max(np.linalg.norm(vh), 1e-8)
        hybrid[w] = vh.tolist()

    sample_word = next(iter(hybrid))
    dim = len(hybrid[sample_word])
    print(f"   결과: {len(hybrid):,} 단어, dim={dim}")

    print(f"\n[3/3] 저장: {out.relative_to(REPO_ROOT)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(hybrid, f, ensure_ascii=False)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"   완료: {size_mb:.1f} MB")

    print("\n다음:")
    print("  → notebooks 에서 spot-check (다른 모델과 비교)")
    print(f"  → 적합하면 swap: cp {out.relative_to(REPO_ROOT)} data/embedding_dictionary_e5.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
