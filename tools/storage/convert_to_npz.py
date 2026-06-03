"""
tools/storage/convert_to_npz.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
무엇 (What)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
임베딩 사전 JSON (1.37GB) → single .npz binary (~250-300MB) 변환. 같은
정보, 약 80% 디스크 절약. 정확도 손실 없음 (float32 그대로).

왜 (Why)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JSON 의 number text encoding 이 비효율적 — 각 float 마다 20+ chars.
binary float32 직접 저장 = 4 bytes/float. 60k × 1024 dim × 4 = 245MB.
실험 임베딩 여러 개 쌓이면 디스크 부족 발생 (PR #22 이후 사용자 통증).

방법 (How)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- np.savez (un-compressed): 빠른 load, ~250MB
- np.savez_compressed: 약 150MB, load 약간 느림 (1-2초 추가)
- 한 .npz 안에 words (object array) + vectors (float32 N×D) 같이

embeddings.py 가 .npz 자동 감지 + 로드. app.py 가 .npz 우선 default.

사용 (Usage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 현 메인 사전 → npz (default: 압축 안 함)
    python tools/storage/convert_to_npz.py

    # 압축 옵션 (더 작음, load 약간 느림)
    python tools/storage/convert_to_npz.py --compress

    # 임의 source / output
    python tools/storage/convert_to_npz.py \\
      --source data/embedding_dictionary_e5.json \\
      --output data/embedding_dictionary_e5.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description="JSON 임베딩 사전 → NPZ 변환")
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA_DIR / "embedding_dictionary_e5.json",
        help="입력 JSON (default: data/embedding_dictionary_e5.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "embedding_dictionary_e5.npz",
        help="출력 NPZ (default: data/embedding_dictionary_e5.npz)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="np.savez_compressed 사용 (더 작음 ~150MB, load 1-2초 추가)",
    )
    args = parser.parse_args()

    src = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    out = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    if not src.exists():
        print(f"❌ source 없음: {src}")
        return 1

    src_mb = src.stat().st_size / 1024 / 1024
    print("=" * 60)
    print(f"JSON → NPZ 변환")
    print("=" * 60)
    print(f"\nsource: {src.relative_to(REPO_ROOT)} ({src_mb:.1f} MB)")

    print(f"\n[1/3] JSON 로드 + 파싱")
    with open(src, "rb") as f:
        d = json.loads(f.read())
    words = list(d.keys())
    matrix = np.array([d[w] for w in words], dtype=np.float32)
    print(f"   단어 수: {len(words):,}")
    print(f"   dim: {matrix.shape[1]}")
    print(f"   matrix size (float32): {matrix.nbytes / 1024 / 1024:.1f} MB")

    print(f"\n[2/3] NPZ 저장" + (" (compressed)" if args.compress else ""))
    out.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if args.compress else np.savez
    saver(out, words=np.array(words, dtype=object), vectors=matrix)

    out_mb = out.stat().st_size / 1024 / 1024
    ratio = (1 - out_mb / src_mb) * 100
    print(f"\n[3/3] 결과")
    print(f"   output: {out.relative_to(REPO_ROOT)} ({out_mb:.1f} MB)")
    print(f"   절약: {src_mb - out_mb:.1f} MB ({ratio:.1f}% ↓)")

    print(f"\n다음 단계:")
    print(f"  1. 게임 테스트 — npz 자동 감지 (.npz 있으면 우선 로드):")
    print(f"     lsof -ti:7860 | xargs kill -9 2>/dev/null; python src/app.py")
    print(f"     로그에 'Loading embeddings from .../e5.npz' 확인")
    print(f"  2. 정상 작동 시 옛 JSON 삭제로 추가 회수:")
    print(f"     rm {src.relative_to(REPO_ROOT)}    # {src_mb:.0f} MB 회수")
    print(f"  3. HF 재업로드 (외부 배포 시):")
    print(f"     hf upload leo4study/ccomantle-embeddings \\")
    print(f"       ./{out.relative_to(REPO_ROOT)} --repo-type dataset")
    print(f"     # EMBEDDING_HF_FILE=embedding_dictionary_e5.npz 환경변수 갱신")
    return 0


if __name__ == "__main__":
    sys.exit(main())
