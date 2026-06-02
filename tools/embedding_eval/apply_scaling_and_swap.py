"""
tools/embedding_eval/apply_scaling_and_swap.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
무엇 (What)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
대안 임베딩 JSON (예: KoE5 raw, build_alt_embeddings.py 산출) 에 mean centering +
uniform scaling 적용해서 ccomantle 게임용 사전 형태로 만든 후 메인 사전과 swap.

왜 (Why)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KoE5 raw 의 cosine 값이 0.7-0.9 사이로 압축돼 있어서 그대로 게임에 쓰면
점수 차이가 흐려짐. E5_embedding_ver2.py 의 scale_embeddings (mean center +
scale 탐색으로 TOP1000≈0.63 만드는 로직) 를 그대로 KoE5 raw 에 적용해서
ccomantle 게임이 기대하는 분포로 변환.

자동 백업 후 swap → 게임 즉시 새 사전 사용.

사용 (Usage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # KoE5 raw → scaling 적용 → 메인 사전 swap
    python tools/embedding_eval/apply_scaling_and_swap.py \\
      --source data/embedding_dictionary_koe5.json

    # dry-run (실제 swap X, scaling 만 시연)
    python tools/embedding_eval/apply_scaling_and_swap.py \\
      --source data/embedding_dictionary_koe5.json --dry-run

산출
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- data/embedding_dictionary_e5.backup_swap_YYYY-MM-DD-HHMMSS.json — 자동 백업
- data/embedding_dictionary_e5.json — KoE5 scaled 로 갱신
- data/embedding_dictionary_<source>_scaled.json — KoE5 scaled 영구 보존 (별도)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SRC_DIR = REPO_ROOT / "src"

# E5_embedding_ver2 의 scale_embeddings 재사용
sys.path.insert(0, str(SRC_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="대안 임베딩에 scaling 적용 + 메인 사전 swap")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="raw 대안 임베딩 JSON 경로 (예: data/embedding_dictionary_koe5.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 swap X — scaling 만 적용 후 통계 출력",
    )
    args = parser.parse_args()

    src_path = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    if not src_path.exists():
        print(f"❌ source 없음: {src_path}")
        return 1

    print("=" * 60)
    print("KoE5 scaling + 메인 사전 swap")
    print("=" * 60)

    # 1. raw 로드
    print(f"\n[1/4] raw 임베딩 로드: {src_path.relative_to(REPO_ROOT)}")
    with open(src_path, "rb") as f:
        raw_dict = json.loads(f.read())
    print(f"   단어 수: {len(raw_dict):,}")

    # 2. scaling 적용 (E5_embedding_ver2 의 scale_embeddings 재사용)
    print(f"\n[2/4] scaling 적용 (mean center + scale 탐색으로 TOP1000≈0.63)")
    try:
        from E5_embedding_ver2 import scale_embeddings
    except ImportError as e:
        print(f"❌ E5_embedding_ver2 import 실패: {e}")
        print("   src/ 에서 import 되는지 확인. pip install -r requirements-dev.txt 도 확인")
        return 1

    scaled_dict = scale_embeddings(raw_dict)
    print(f"   scaling 완료: {len(scaled_dict):,} 단어")

    # 3. scaled 영구 저장 (별 이름으로)
    source_name = src_path.stem.replace("embedding_dictionary_", "")  # koe5
    scaled_path = DATA_DIR / f"embedding_dictionary_{source_name}_scaled.json"
    with open(scaled_path, "w", encoding="utf-8") as f:
        json.dump(scaled_dict, f, ensure_ascii=False)
    print(f"\n[3/4] scaled 영구 저장: {scaled_path.relative_to(REPO_ROOT)}")

    if args.dry_run:
        print("\n--dry-run: 실제 swap 건너뜀. 메인 사전 (embedding_dictionary_e5.json) 변동 X.")
        print(f"   필요 시 수동 swap: cp {scaled_path.relative_to(REPO_ROOT)} data/embedding_dictionary_e5.json")
        return 0

    # 4. 메인 사전 백업 + swap
    main_path = DATA_DIR / "embedding_dictionary_e5.json"
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    backup_path = DATA_DIR / f"embedding_dictionary_e5.backup_swap_{stamp}.json"

    print(f"\n[4/4] 메인 사전 백업 + swap")
    if main_path.exists():
        shutil.copy(main_path, backup_path)
        print(f"   📦 백업: {backup_path.relative_to(REPO_ROOT)}")

    shutil.copy(scaled_path, main_path)
    print(f"   ✓ swap 완료: {main_path.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 60)
    print("완료")
    print("=" * 60)
    print(f"  source 모델: {source_name}")
    print(f"  메인 사전 → {source_name} scaled 로 교체됨")
    print(f"  옛 사전 백업: {backup_path.name}")
    print()
    print("다음 단계:")
    print("  1. 서버 재시작 후 게임 테스트:")
    print("     lsof -ti:7860 | xargs kill -9 2>/dev/null; python src/app.py")
    print("  2. 정답 '사과' 설정 후 과일 시도 — 자연스러운 점수 분포 확인")
    print("  3. 만족 시 HF 재업로드:")
    print("     hf upload leo4study/ccomantle-embeddings \\")
    print("       ./data/embedding_dictionary_e5.json --repo-type dataset")
    print("  4. 점수 이상 시 백업 복원:")
    print(f"     cp {backup_path.relative_to(REPO_ROOT)} data/embedding_dictionary_e5.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
