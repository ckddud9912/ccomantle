"""
tools/dict_quality/coverage_check.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
무엇 (What)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
mecab-ko-dic 의 일반명사(NNG) 표제어와 ccomantle 의 5만 단어 사전을 비교해
(1) coverage, (2) 누락 단어 후보, (3) 필터 거부 분류 산출.

왜 (Why)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
docs/features/05_evaluation_methodology.md §1 의 진단 도구. 사용자가 게임에서
"사전에 없는 단어" 거부를 겪을 때 어떤 한국어 명사가 빠져있는지 정량 식별.

방법론·이유는 docs/features/05_evaluation_methodology.md §1 참조.

사용 (Usage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # mecab-ko-dic 자동 다운로드 (첫 실행 시) + 비교
    python tools/dict_quality/coverage_check.py

    # 이미 받아둔 mecab-ko-dic 경로 직접 지정
    python tools/dict_quality/coverage_check.py --mecab-dir /path/to/mecab-ko-dic

산출 (Output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- data/quality_report.json — coverage 통계 + 필터 거부 분류
- data/missing_words_candidates.json — 누락 단어 top-N (cost 오름차순, 우선순위 순)
- stdout — 요약 + top 20 누락 단어
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
REFS_DIR = Path(__file__).parent / "refs"

# mecab-ko-dic 공식 배포 (Apache 2.0)
MECAB_VERSION = "2.1.1-20180720"
MECAB_URL = f"https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-{MECAB_VERSION}.tar.gz"

# make_words_from_vec.py 의 필터와 동일 규칙 (시뮬레이션용)
KOREAN_PATTERN = re.compile(r"^[가-힣]+$")
ADVERB_PATTERN = re.compile(r".+(게|히)$")
MAX_RANK_CAP = 50_000


# ─────────────────────────────────────────────────────────────────
# mecab-ko-dic 다운로드 + 파싱
# ─────────────────────────────────────────────────────────────────

def ensure_mecab_dic() -> Path:
    """mecab-ko-dic 자동 다운로드 + 압축 해제. 이미 있으면 skip.

    Returns:
        압축 해제된 디렉토리 경로.
    """
    target = REFS_DIR / f"mecab-ko-dic-{MECAB_VERSION}"
    if target.exists():
        return target

    REFS_DIR.mkdir(parents=True, exist_ok=True)
    tarball = REFS_DIR / f"mecab-ko-dic-{MECAB_VERSION}.tar.gz"

    if not tarball.exists():
        print(f"📥 다운로드 중: {MECAB_URL}")
        urllib.request.urlretrieve(MECAB_URL, tarball)
        print(f"   → {tarball.stat().st_size / 1024 / 1024:.1f} MB 수신")

    print(f"📦 압축 해제: {tarball.name}")
    with tarfile.open(tarball) as tf:
        tf.extractall(REFS_DIR)

    if not target.exists():
        raise RuntimeError(
            f"압축 해제 후 예상 경로 없음: {target}\n"
            f"REFS_DIR 내용: {list(REFS_DIR.iterdir())}"
        )
    return target


def parse_nng(mecab_dir: Path) -> dict[str, int]:
    """NNG.csv → {표제어: cost}. cost 낮을수록 빈도·우선순위 높음.

    Args:
        mecab_dir: 압축 해제된 mecab-ko-dic 디렉토리.

    Returns:
        {word: min_cost}. 같은 단어가 여러 row 면 최소 cost 채택.
    """
    nng_path = mecab_dir / "NNG.csv"
    if not nng_path.exists():
        raise FileNotFoundError(f"NNG.csv 없음: {nng_path}")

    # mecab-ko-dic 의 CSV 는 UTF-8. 일부 옛 버전이 EUC-KR 인 경우 폴백.
    try:
        text = nng_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = nng_path.read_text(encoding="euc-kr")

    nouns: dict[str, int] = {}
    for line in text.splitlines():
        cols = line.split(",")
        if len(cols) < 5:
            continue
        word = cols[0].strip()
        try:
            cost = int(cols[3])
        except ValueError:
            continue
        if not word:
            continue
        # 중복 표제어 → 최소 cost (가장 빈도 높은 형태)
        if word not in nouns or cost < nouns[word]:
            nouns[word] = cost

    return nouns


# ─────────────────────────────────────────────────────────────────
# ccomantle 사전 + 필터 시뮬레이션
# ─────────────────────────────────────────────────────────────────

def load_ccomantle_dict() -> set[str]:
    """ccomantle 의 실제 사전 — game.py 가 로드하는 embedding_dictionary_e5.json 의 키.

    words_*.json 은 빌드 input 일 뿐 게임이 직접 안 봄. expand_dict.py 같은 작업으로
    임베딩 JSON 만 갱신되고 words_*.json 이 옛 상태로 남는 경우가 있어서, 실제
    게임 사전 (= embedding dict 의 키) 을 진단 기준으로 사용.
    """
    path = DATA_DIR / "embedding_dictionary_e5.json"
    if not path.exists():
        raise FileNotFoundError(
            f"ccomantle 임베딩 사전 없음: {path}\n"
            f"먼저 python src/E5_embedding_ver2.py 로 생성하거나 HF 에서 받아두세요."
        )
    # 키만 필요 (벡터는 read 만 해서 메모리 낭비 — orjson 으로 빠르게)
    with open(path, "rb") as f:
        emb_dict = json.loads(f.read())
    return set(emb_dict.keys())


def filter_rejection_reason(word: str) -> str:
    """make_words_from_vec.py 의 is_valid_word 와 동일 규칙으로 거부 이유 분류.

    Returns:
        "not_pure_korean" / "too_short" / "too_long" / "adverb_pattern" /
        "passes_filter" — 마지막 경우는 cap 50k 초과 또는 FastText 부재가 원인.
    """
    if not KOREAN_PATTERN.fullmatch(word):
        return "not_pure_korean"
    if len(word) < 2:
        return "too_short"
    if len(word) > 6:
        return "too_long"
    if ADVERB_PATTERN.fullmatch(word):
        return "adverb_pattern"
    return "passes_filter"  # → cap 또는 FastText 부재 (현 PR 에선 구분 X)


# ─────────────────────────────────────────────────────────────────
# 비교 + 산출
# ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="mecab-ko-dic NNG vs ccomantle 사전 비교")
    parser.add_argument(
        "--mecab-dir",
        type=Path,
        default=None,
        help="이미 다운로드한 mecab-ko-dic 디렉토리 (지정 시 자동 다운로드 skip)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5000,
        help="저장할 누락 단어 top N (cost 오름차순, 기본 5000)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ccomantle 사전 coverage 검사 — mecab-ko-dic NNG 기준")
    print("=" * 60)

    # 1. mecab-ko-dic 준비
    print("\n[1/4] mecab-ko-dic 준비")
    mecab_dir = args.mecab_dir if args.mecab_dir else ensure_mecab_dic()
    print(f"   사용: {mecab_dir}")

    # 2. NNG 파싱
    print("\n[2/4] NNG.csv 파싱")
    nng = parse_nng(mecab_dir)
    print(f"   NNG 표제어 (중복 제거 후): {len(nng):,}")

    # 3. ccomantle 사전 로드
    print("\n[3/4] ccomantle 사전 로드")
    ccm = load_ccomantle_dict()
    print(f"   ccomantle 단어: {len(ccm):,}")

    # 4. 비교
    print("\n[4/4] 비교 + 분석")
    nng_set = set(nng.keys())
    overlap = nng_set & ccm
    missing = nng_set - ccm
    coverage = len(overlap) / len(nng_set) if nng_set else 0.0

    # 누락 단어 필터 거부 이유 분류
    breakdown: Counter[str] = Counter()
    for w in missing:
        breakdown[filter_rejection_reason(w)] += 1

    # 우선순위 정렬 — cost 오름차순 (빈도 높은 단어 우선)
    missing_sorted = sorted(missing, key=lambda w: nng[w])

    # ─────────────────────────────────────────────────────────────
    # 결과 출력
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"ccomantle 사전:        {len(ccm):>8,}")
    print(f"mecab NNG:             {len(nng_set):>8,}")
    print(f"교집합 (이미 있음):    {len(overlap):>8,}")
    print(f"Coverage (NNG ∩ ccm / NNG): {coverage:>6.2%}")
    print(f"누락 수:               {len(missing):>8,}")

    print("\n누락 단어 필터 거부 원인 분류:")
    for reason, count in sorted(breakdown.items(), key=lambda x: -x[1]):
        pct = count / len(missing) * 100 if missing else 0
        print(f"   {reason:<24}: {count:>8,}  ({pct:>5.1f}%)")
    print("   ※ passes_filter 는 ccomantle 필터를 통과하지만 빈도 cap 50k 초과 또는 FastText 부재")

    print("\nTop 20 누락 (cost 낮은 순 = 빈도 높은 추정):")
    for i, w in enumerate(missing_sorted[:20], 1):
        passes = filter_rejection_reason(w) == "passes_filter"
        marker = "✓" if passes else "✗"
        print(f"   {i:>2}. {w:<10} cost={nng[w]:>5} {marker}{'필터 통과' if passes else '필터 거부'}")

    # ─────────────────────────────────────────────────────────────
    # JSON 저장
    # ─────────────────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    quality_report = {
        "generated_at": datetime.now().isoformat(),
        "ccomantle_size": len(ccm),
        "mecab_nng_size": len(nng_set),
        "overlap": len(overlap),
        "coverage": round(coverage, 4),
        "missing_count": len(missing),
        "filter_rejection_breakdown": dict(breakdown),
        "mecab_version": MECAB_VERSION,
    }
    report_path = DATA_DIR / "quality_report.json"
    report_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    candidates = [
        {
            "word": w,
            "cost": nng[w],
            "mecab_rank": i + 1,
            "passes_ccomantle_filter": filter_rejection_reason(w) == "passes_filter",
            "rejection_reason": filter_rejection_reason(w),
        }
        for i, w in enumerate(missing_sorted[: args.top_n])
    ]
    candidates_path = DATA_DIR / "missing_words_candidates.json"
    candidates_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("저장 완료")
    print("=" * 60)
    print(f"  📄 {report_path.relative_to(REPO_ROOT)}")
    print(f"  📄 {candidates_path.relative_to(REPO_ROOT)} (top {args.top_n:,})")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
