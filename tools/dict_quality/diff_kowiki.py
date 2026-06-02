"""
tools/dict_quality/diff_kowiki.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
무엇 (What)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
한국어 위키피디아의 article title dump 와 ccomantle 사전을 비교해 누락 단어
식별. mecab-ko-dic 보다 동시대 어휘 (외래어·고유명사·신어 포함) 가 풍부함.

왜 (Why)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PR #15·#17 의 mecab-ko-dic (2018-07-20) 기반 진단·보강이 "최신성" 결에서
사용자 통증 (동시대 일반 명사 누락) 과 미스매치. ko 위키 title dump 는 2주마다
갱신되어 동시대 어휘 직격. 본 스크립트가 그 source 의 diff 산출.

방법 (How)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ko 위키 latest titles dump (ns0 = articles) 자동 다운로드 (~30MB gz)
2. 필터: 한글-only / 길이 1-6 / 단일 단어 (공백 없음)
3. mecab-ko-dic NNG.csv 와 cross-ref → 우선순위 점수 부여
   - mecab 에 있으면 cost 값 (낮을수록 빈도 ↑)
   - ko 위키 only 면 "modern_only" 마크 (동시대 어휘 또는 고유명사)
4. ccomantle 사전 (embedding_dictionary_e5.json 키) 과 diff → 누락 식별
5. JSON 저장 + 요약 출력

사용 (Usage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python tools/dict_quality/diff_kowiki.py

    # 옵션
    python tools/dict_quality/diff_kowiki.py --top-n 10000
    python tools/dict_quality/diff_kowiki.py --no-mecab-crossref

산출
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- data/missing_words_candidates_kowiki.json — 누락 단어 (mecab cost 있으면 정렬 우선, 없으면 alphabetical)
- data/quality_report_kowiki.json — coverage 통계 + source 분류
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
REFS_DIR = Path(__file__).parent / "refs"

# ko 위키 latest titles dump (namespace 0 = articles only)
KOWIKI_URL = "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-all-titles-in-ns0.gz"

# mecab-ko-dic (이전 PR 에서 받아둔 거 재사용)
MECAB_VERSION = "2.1.1-20180720"

# ccomantle 의 추출 필터와 동일 (1글자 허용 — PR #16 의 길이 cutoff)
KOREAN_PATTERN = re.compile(r"^[가-힣]+$")
LEN_MIN = 1
LEN_MAX = 6


def ensure_kowiki_titles() -> Path:
    """ko 위키 titles dump 다운로드. 이미 있으면 skip."""
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = REFS_DIR / "kowiki-latest-all-titles-in-ns0.gz"

    if gz_path.exists():
        size_mb = gz_path.stat().st_size / 1024 / 1024
        print(f"   기존 사용: {gz_path.name} ({size_mb:.1f} MB)")
        return gz_path

    print(f"📥 다운로드 중: {KOWIKI_URL}")
    urllib.request.urlretrieve(KOWIKI_URL, gz_path)
    size_mb = gz_path.stat().st_size / 1024 / 1024
    print(f"   → {size_mb:.1f} MB 수신")
    return gz_path


def parse_kowiki_titles(gz_path: Path) -> set[str]:
    """gz 파일에서 한 줄 한 단어 (단일 한글, 길이 1-6) 만 추출."""
    titles: set[str] = set()
    total = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        first = f.readline()  # 첫 줄: 헤더 "page_title"
        if "page_title" not in first.lower():
            # 첫 줄도 title 이면 다시 처리
            t = first.strip()
            if _is_valid_title(t):
                titles.add(t)
                total += 1

        for line in f:
            total += 1
            t = line.strip()
            if _is_valid_title(t):
                titles.add(t)

    print(f"   ko 위키 title 총 {total:,} 중 필터 통과: {len(titles):,}")
    return titles


def _is_valid_title(title: str) -> bool:
    """단일 한글 단어 (공백/특수문자 없음, 길이 1-6) 만 통과."""
    if not title:
        return False
    if not KOREAN_PATTERN.fullmatch(title):
        return False
    if not (LEN_MIN <= len(title) <= LEN_MAX):
        return False
    return True


def load_mecab_nng_costs() -> dict[str, int]:
    """이전 PR 에서 받아둔 mecab NNG.csv → {표제어: cost}. 없으면 빈 dict."""
    mecab_dir = REFS_DIR / f"mecab-ko-dic-{MECAB_VERSION}"
    nng_path = mecab_dir / "NNG.csv"
    if not nng_path.exists():
        print(f"   mecab-ko-dic 없음 (cost cross-ref skip): {nng_path}")
        return {}

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
        if word not in nouns or cost < nouns[word]:
            nouns[word] = cost
    print(f"   mecab NNG cost cross-ref: {len(nouns):,} 단어")
    return nouns


def load_ccomantle_dict() -> set[str]:
    """게임이 실제로 로드하는 embedding_dictionary_e5.json 의 키."""
    path = DATA_DIR / "embedding_dictionary_e5.json"
    if not path.exists():
        raise FileNotFoundError(f"ccomantle 사전 없음: {path}")
    with open(path, "rb") as f:
        return set(json.loads(f.read()).keys())


def main() -> int:
    parser = argparse.ArgumentParser(description="ko 위키 titles vs ccomantle diff")
    parser.add_argument("--top-n", type=int, default=10000, help="저장할 누락 단어 top N")
    parser.add_argument("--no-mecab-crossref", action="store_true", help="mecab cost cross-ref skip")
    args = parser.parse_args()

    print("=" * 60)
    print("ccomantle vs ko 위키 titles diff")
    print("=" * 60)

    # 1. ko 위키 titles 준비
    print("\n[1/5] ko 위키 titles dump 준비")
    gz_path = ensure_kowiki_titles()

    # 2. titles 파싱
    print("\n[2/5] titles 파싱 (한글-only, 길이 1-6)")
    kowiki_titles = parse_kowiki_titles(gz_path)

    # 3. mecab cost cross-ref (선택)
    if args.no_mecab_crossref:
        print("\n[3/5] mecab cost cross-ref skip (--no-mecab-crossref)")
        mecab_costs: dict[str, int] = {}
    else:
        print("\n[3/5] mecab NNG cost cross-ref 로드")
        mecab_costs = load_mecab_nng_costs()

    # 4. ccomantle 사전 로드
    print("\n[4/5] ccomantle 사전 로드")
    ccm = load_ccomantle_dict()
    print(f"   ccomantle 단어: {len(ccm):,}")

    # 5. 비교 + 분석
    print("\n[5/5] diff + 분석")
    overlap = kowiki_titles & ccm
    missing = kowiki_titles - ccm
    coverage = len(overlap) / len(kowiki_titles) if kowiki_titles else 0.0

    # 누락 단어 분류 + 정렬
    in_mecab = [w for w in missing if w in mecab_costs]
    modern_only = [w for w in missing if w not in mecab_costs]  # ko 위키만 — 동시대/고유명사

    in_mecab.sort(key=lambda w: mecab_costs[w])  # cost 낮은 순
    modern_only.sort()  # alphabetical (빈도 정보 X)

    # ─────────────────────────────────────────────────
    # 결과 출력
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"ccomantle 사전:          {len(ccm):>8,}")
    print(f"ko 위키 한글 titles:     {len(kowiki_titles):>8,}")
    print(f"교집합 (이미 있음):      {len(overlap):>8,}")
    print(f"Coverage:                {coverage:>7.2%}")
    print(f"누락 수:                 {len(missing):>8,}")
    print()
    print("누락 단어 source 분류:")
    print(f"   mecab 에도 있음 (전통 어휘):   {len(in_mecab):>8,}")
    print(f"   ko 위키 only (modern/고유명사): {len(modern_only):>8,}")

    print("\nTop 30 누락 — mecab 에도 있음 (cost 낮은 순 = 빈도 추정 ↑):")
    for i, w in enumerate(in_mecab[:30], 1):
        print(f"   {i:>2}. {w:<12} mecab_cost={mecab_costs[w]:>5}")

    print("\nTop 30 누락 — ko 위키 only (modern/고유명사):")
    for i, w in enumerate(modern_only[:30], 1):
        print(f"   {i:>2}. {w}")

    # ─────────────────────────────────────────────────
    # JSON 저장
    # ─────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    quality_report = {
        "generated_at": datetime.now().isoformat(),
        "source": "kowiki-latest-all-titles-in-ns0",
        "ccomantle_size": len(ccm),
        "kowiki_korean_titles": len(kowiki_titles),
        "overlap": len(overlap),
        "coverage": round(coverage, 4),
        "missing_count": len(missing),
        "missing_in_mecab": len(in_mecab),
        "missing_modern_only": len(modern_only),
    }
    report_path = DATA_DIR / "quality_report_kowiki.json"
    report_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # candidates: in_mecab 먼저 (cost 정렬), 그 뒤 modern_only (alphabetical)
    candidates = []
    for w in in_mecab[: args.top_n]:
        candidates.append({"word": w, "cost": mecab_costs[w], "source": "kowiki+mecab"})

    remaining = args.top_n - len(candidates)
    if remaining > 0:
        for w in modern_only[:remaining]:
            candidates.append({"word": w, "cost": None, "source": "kowiki_only"})

    candidates_path = DATA_DIR / "missing_words_candidates_kowiki.json"
    candidates_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("저장 완료")
    print("=" * 60)
    print(f"  📄 {report_path.relative_to(REPO_ROOT)}")
    print(f"  📄 {candidates_path.relative_to(REPO_ROOT)} (top {args.top_n:,})")
    print()
    print("다음 단계:")
    print("  1. 산출된 candidates JSON spot-check (어떤 단어들이 들어있나)")
    print("  2. 필요시 expand_dict.py 를 후보 파일 옵션 받게 확장 후 보강 진행")
    print("  3. 우리말샘 dump 도착 시 합집합 reference 로 정확도 ↑")

    return 0


if __name__ == "__main__":
    sys.exit(main())
