"""
tools/dict_quality/expand_dict.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
무엇 (What)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
coverage_check.py 가 만든 missing_words_candidates.json 의 단어들을 현 사전에
추가. FastText .vec 재다운로드 없이 누락된 한국어 명사 (top N, default 5000)
보강.

왜 (Why)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
docs/features/05_evaluation_methodology.md §1.7 의 finding 1·2·3 의 누락 단어
("끝", "꿈", "눈", "가게", "무게", "학년도", "큰일" 등) 을 게임 사전에 즉시
추가. handoff.md 의 사용자 통증 ("사전에 없는 단어" 거부) 직접 해소.

방법 (How — distribution 보존 path)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 기존 embedding_dictionary_e5.json (50k scaled) 로드
2. missing_words_candidates.json 의 top N 새 단어 추출 (기존 단어와 중복 제거)
3. 새 단어들만 E5-large 로 임베딩 생성 (raw, unit-norm) — MPS 로 5분 내외
4. 그대로 머지. scale_embeddings 재실행 X — 기존 50k 의 scaled 분포 보존
5. 게임 점수의 distribution 보정은 game.py 의 동적 sim_alpha 가 매 정답마다 흡수
6. 결과 저장 (in-place, 자동 백업)

**왜 scale 재실행 안 하는가**: 기존 50k 가 이미 mean-centered + scaled 임베딩.
mean-centering 을 한 번 더 적용하면 기존 단어들 cosine 분포가 압축돼서 다수
점수가 0 근처/음수가 됨 → game._scale_scalar 가 음수는 0 반환 → 점수 망가짐.
새 단어 raw 의 분포 차이는 sim_alpha 동적 보정으로 충분.

사용 (Usage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 사전 준비
    pip install -r requirements-dev.txt    # torch + transformers + tqdm

    # 기본: missing_words_candidates.json top 5000 추가
    python tools/dict_quality/expand_dict.py

    # top N 변경
    python tools/dict_quality/expand_dict.py --top-n 1500

    # 드라이런 (실제 저장 X)
    python tools/dict_quality/expand_dict.py --dry-run

산출
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- data/embedding_dictionary_e5.json — in-place 갱신
- data/embedding_dictionary_e5.backup_YYYY-MM-DD-HHMMSS.json — 자동 백업
- data/words_NNNNN.json — 새 단어 리스트 (확장된 크기 반영)
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

# E5_embedding_ver2 의 함수들을 재사용
sys.path.insert(0, str(SRC_DIR))


def load_existing_embeddings(path: Path) -> dict[str, list[float]]:
    """기존 embedding_dictionary_e5.json 로드. scaled 임베딩 가정."""
    if not path.exists():
        raise FileNotFoundError(f"기존 사전 없음: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_missing_candidates(path: Path, top_n: int) -> list[str]:
    """missing_words_candidates.json 의 top N 단어 추출."""
    if not path.exists():
        raise FileNotFoundError(
            f"누락 후보 파일 없음: {path}\n"
            f"먼저 python tools/dict_quality/coverage_check.py 실행하세요."
        )
    with open(path, encoding="utf-8") as f:
        cands = json.load(f)
    return [c["word"] for c in cands[:top_n]]


def main() -> int:
    parser = argparse.ArgumentParser(description="ccomantle 사전 어휘 보강")
    parser.add_argument(
        "--top-n",
        type=int,
        default=5000,
        help="missing_words_candidates.json 에서 추가할 단어 수 (default 5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 저장 X, 분석만 출력",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ccomantle 사전 어휘 보강 (expand)")
    print("=" * 60)

    embedding_path = DATA_DIR / "embedding_dictionary_e5.json"
    candidates_path = DATA_DIR / "missing_words_candidates.json"

    # ─────────────────────────────────────────────────────────
    # 1. 기존 / 후보 로드
    # ─────────────────────────────────────────────────────────
    print("\n[1/5] 기존 사전 + 누락 후보 로드")
    existing = load_existing_embeddings(embedding_path)
    print(f"   기존 단어: {len(existing):,}")

    candidates = load_missing_candidates(candidates_path, args.top_n)
    print(f"   누락 후보 (top {args.top_n}): {len(candidates):,}")

    # 중복 제거 — candidates 중 이미 existing 에 있으면 skip
    existing_set = set(existing.keys())
    new_words = [w for w in candidates if w not in existing_set]
    duplicates = len(candidates) - len(new_words)
    print(f"   기존과 중복: {duplicates:,}")
    print(f"   실제 추가될 단어: {len(new_words):,}")

    if not new_words:
        print("\n추가할 단어 없음. 종료.")
        return 0

    print(f"\n   샘플 (앞 10개): {new_words[:10]}")

    if args.dry_run:
        print("\n--dry-run: 여기서 종료. 실제 임베딩/저장 안 함.")
        return 0

    # ─────────────────────────────────────────────────────────
    # 2. E5 모델 로드
    # ─────────────────────────────────────────────────────────
    print("\n[2/5] E5 모델 로드")
    try:
        from E5_embedding_ver2 import (
            load_e5_model,
            build_embedding_dict,
            save_embedding_dict,
            MODEL_NAME,
        )
    except ImportError as e:
        print(f"\n❌ E5_embedding_ver2 import 실패: {e}")
        print("   pip install -r requirements-dev.txt 먼저 실행하세요.")
        return 1

    tokenizer, model = load_e5_model(MODEL_NAME)

    # ─────────────────────────────────────────────────────────
    # 3. 새 단어 임베딩 생성 (raw, unit-norm)
    # ─────────────────────────────────────────────────────────
    print(f"\n[3/5] 새 단어 {len(new_words):,}개 E5 임베딩 생성")
    new_emb = build_embedding_dict(new_words, tokenizer, model)
    print(f"   완료: {len(new_emb):,}개")

    # ─────────────────────────────────────────────────────────
    # 4. 머지 (scale 재실행 X — 기존 50k 분포 보존)
    # ─────────────────────────────────────────────────────────
    print(f"\n[4/5] 머지 (scale 재실행 X — 기존 분포 보존)")
    merged: dict[str, list[float]] = {**existing, **new_emb}
    print(f"   머지 총: {len(merged):,}")
    print(f"   기존 50k 의 scaled 분포 그대로 유지")
    print(f"   새 {len(new_emb):,}k 는 E5 raw (unit-norm) 그대로 추가")
    print(f"   game.py 의 동적 sim_alpha 가 distribution 차이 흡수")
    scaled = merged  # 호환: 이후 코드에서 동일 변수명 사용

    # ─────────────────────────────────────────────────────────
    # 5. 저장 (자동 백업)
    # ─────────────────────────────────────────────────────────
    print(f"\n[5/5] 저장 (자동 백업)")

    # 백업
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    backup_path = DATA_DIR / f"embedding_dictionary_e5.backup_{stamp}.json"
    shutil.copy(embedding_path, backup_path)
    print(f"   📦 백업: {backup_path.relative_to(REPO_ROOT)}")

    # in-place 갱신
    save_embedding_dict(str(embedding_path), scaled)
    print(f"   ✓ 갱신: {embedding_path.relative_to(REPO_ROOT)}")

    # 단어 리스트 갱신
    new_words_path = DATA_DIR / f"words_{len(scaled)}.json"
    with open(new_words_path, "w", encoding="utf-8") as f:
        json.dump(list(scaled.keys()), f, ensure_ascii=False, indent=2)
    print(f"   ✓ 단어 리스트: {new_words_path.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 60)
    print("완료")
    print("=" * 60)
    print(f"  기존: {len(existing):,} 단어")
    print(f"  추가: {len(new_words):,} 단어")
    print(f"  최종: {len(scaled):,} 단어")
    print()
    print("다음 단계:")
    print("  1. python tools/dict_quality/coverage_check.py  # coverage 변화 확인")
    print("  2. 게임 실행 + 누락됐던 단어 ('끝'/'꿈'/'가게'/'학년도') 직접 테스트")
    print("  3. HF dataset 업로드 (운영자 leo4study):")
    print("     huggingface-cli upload leo4study/ccomantle-embeddings \\")
    print("       ./data/embedding_dictionary_e5.json --repo-type dataset")

    return 0


if __name__ == "__main__":
    sys.exit(main())
