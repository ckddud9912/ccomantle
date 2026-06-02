"""
tools/dict_quality/diff_urimalsaem.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
무엇 (What)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
국립국어원 우리말샘 bulk dump (25 chunk, ~1.2M entry) 와 ccomantle 사전을
비교해 누락 단어를 POS 별로 분류 식별. mecab/위키 보다 공식·최신·포괄적
한국어 사전 source.

왜 (Why)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PR #15·#17 의 mecab 2018 기반 보강이 사용자 진짜 통증 (동시대 일반 명사
누락) 과 미스매치. 우리말샘 = 국립국어원 공식 + 분기별 갱신 + 신어 포함 →
가장 정확한 reference. POS 별로 누락 분류하면 동사·형용사 정책 결정에도
도움.

방법 (How)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. tools/dict_quality/refs/urimalsaem_2026-05-03/ 의 25 chunk 순차 로드
2. 각 entry 필터:
   - senseinfo.type == "일반어" 만 (방언·북한어·옛말·외래어 제외)
   - 표제어 하이픈 제거: "겁-쟁이" → "겁쟁이"
   - 한글-only, 길이 1-6
3. POS 별 set 분류 (명사·동사·형용사·부사·기타)
4. (선택) mecab NNG cost cross-ref → 누락 명사 우선순위 정렬
5. ccomantle 사전과 POS 별 diff
6. 산출 — 터미널 요약만, 상세는 파일로:
   - data/quality_report_urimalsaem.md  (사용자 읽기용)
   - data/quality_report_urimalsaem.json (머신 읽기용)
   - data/missing_words_urimalsaem_{nouns,verbs,adj,adverbs}.json

사용 (Usage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python tools/dict_quality/diff_urimalsaem.py
    python tools/dict_quality/diff_urimalsaem.py --no-mecab-crossref
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
REFS_DIR = Path(__file__).parent / "refs"
URIMALSAEM_DIR = REFS_DIR / "urimalsaem_2026-05-03"

MECAB_VERSION = "2.1.1-20180720"

KOREAN_PATTERN = re.compile(r"^[가-힣]+$")
LEN_MIN = 1
LEN_MAX = 6

POS_LABELS = {
    "명사": "nouns",
    "동사": "verbs",
    "형용사": "adjectives",
    "부사": "adverbs",
}


def normalize_word(word: str) -> str:
    """우리말샘 표제어 정규화: '겁-쟁이' → '겁쟁이'."""
    return word.replace("-", "").strip()


def is_valid_word(word: str) -> bool:
    if not KOREAN_PATTERN.fullmatch(word):
        return False
    if not (LEN_MIN <= len(word) <= LEN_MAX):
        return False
    return True


def load_mecab_nng_costs() -> dict[str, int]:
    """이전 PR 의 mecab NNG.csv → {표제어: cost}. 명사 cross-ref 용."""
    mecab_dir = REFS_DIR / f"mecab-ko-dic-{MECAB_VERSION}"
    nng_path = mecab_dir / "NNG.csv"
    if not nng_path.exists():
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
    return nouns


def load_ccomantle_dict() -> set[str]:
    path = DATA_DIR / "embedding_dictionary_e5.json"
    if not path.exists():
        raise FileNotFoundError(f"ccomantle 사전 없음: {path}")
    with open(path, "rb") as f:
        return set(json.loads(f.read()).keys())


def process_chunks() -> tuple[dict[str, set[str]], dict[str, int]]:
    """모든 chunk 순차 로드 → POS 별 set + 처리 통계."""
    chunk_paths = sorted(URIMALSAEM_DIR.glob("*.json"))
    if not chunk_paths:
        raise FileNotFoundError(f"우리말샘 chunk 없음: {URIMALSAEM_DIR}")

    pos_sets: dict[str, set[str]] = {
        "nouns": set(),
        "verbs": set(),
        "adjectives": set(),
        "adverbs": set(),
        "others": set(),
    }
    stats = Counter({
        "total_entries": 0,
        "filtered_non_general": 0,  # 일반어 아님 (방언/옛말/북한어 등)
        "filtered_invalid_word": 0,  # 한글 아님 / 길이 초과
        "kept": 0,
    })

    print(f"   chunk {len(chunk_paths)}개 순차 처리 중...")
    for i, p in enumerate(chunk_paths, 1):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        # 우리말샘 dump 가 list 또는 dict 일 수 있음 — 자동 감지
        if isinstance(data, dict):
            # 일부 dump 는 {"channel": {"item": [...]}} 같은 wrapper 가짐
            # 가장 흔한 패턴: dict 안의 가장 큰 list 찾기
            entries = _extract_entries_from_dict(data)
        elif isinstance(data, list):
            entries = data
        else:
            print(f"   [WARN] {p.name} 의 root type 모름: {type(data).__name__}, skip")
            continue

        for entry in entries:
            stats["total_entries"] += 1
            wordinfo = entry.get("wordinfo", {}) if isinstance(entry, dict) else {}
            senseinfo = entry.get("senseinfo", {}) if isinstance(entry, dict) else {}

            if senseinfo.get("type") != "일반어":
                stats["filtered_non_general"] += 1
                continue

            raw_word = wordinfo.get("word", "")
            normalized = normalize_word(raw_word)
            if not is_valid_word(normalized):
                stats["filtered_invalid_word"] += 1
                continue

            stats["kept"] += 1
            pos = senseinfo.get("pos", "")
            key = POS_LABELS.get(pos, "others")
            pos_sets[key].add(normalized)

        if i % 5 == 0 or i == len(chunk_paths):
            print(f"   ... {i}/{len(chunk_paths)} chunk 처리 완료")

    return pos_sets, dict(stats)


def _extract_entries_from_dict(data: dict) -> list:
    """우리말샘 dump 가 wrapper dict 일 때 entry list 추출."""
    # 흔한 후보: data["channel"]["item"], data["item"], data["entries"]
    for path in [("channel", "item"), ("item",), ("entries",), ("data",)]:
        cur = data
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, list):
            return cur
    # 그래도 못 찾으면 dict 의 모든 list value 중 가장 큰 거
    biggest = []
    for v in data.values():
        if isinstance(v, list) and len(v) > len(biggest):
            biggest = v
    return biggest


def write_outputs(
    pos_sets: dict[str, set[str]],
    stats: dict,
    ccm: set[str],
    mecab_costs: dict[str, int],
) -> dict:
    """결과 파일들 저장 + summary dict 반환."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pos_results: dict[str, dict] = {}
    for pos_key, words in pos_sets.items():
        overlap = words & ccm
        missing = words - ccm
        coverage = len(overlap) / len(words) if words else 0.0

        # 명사만 mecab cost cross-ref (다른 POS 는 mecab NNG 와 결 다름)
        if pos_key == "nouns" and mecab_costs:
            in_mecab = sorted(
                [(w, mecab_costs[w]) for w in missing if w in mecab_costs],
                key=lambda x: x[1],
            )
            urimalsaem_only = sorted(missing - set(mecab_costs.keys()))
            missing_sorted = [
                {"word": w, "cost": c, "source": "urimalsaem+mecab"}
                for w, c in in_mecab
            ] + [
                {"word": w, "cost": None, "source": "urimalsaem_only"}
                for w in urimalsaem_only
            ]
        else:
            missing_sorted = [
                {"word": w, "cost": None, "source": "urimalsaem_only"}
                for w in sorted(missing)
            ]

        pos_results[pos_key] = {
            "total": len(words),
            "overlap": len(overlap),
            "missing": len(missing),
            "coverage": round(coverage, 4),
            "missing_list": missing_sorted,
        }

        # POS 별 후보 JSON 저장
        fname_map = {
            "nouns": "missing_words_urimalsaem_nouns.json",
            "verbs": "missing_words_urimalsaem_verbs.json",
            "adjectives": "missing_words_urimalsaem_adj.json",
            "adverbs": "missing_words_urimalsaem_adverbs.json",
            "others": "missing_words_urimalsaem_others.json",
        }
        out_path = DATA_DIR / fname_map[pos_key]
        out_path.write_text(
            json.dumps(missing_sorted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 통합 요약 JSON
    summary = {
        "generated_at": datetime.now().isoformat(),
        "source": "urimalsaem_2026-05-03",
        "ccomantle_size": len(ccm),
        "processing_stats": stats,
        "pos_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "missing_list"}
            for k, v in pos_results.items()
        },
    }
    (DATA_DIR / "quality_report_urimalsaem.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # MD 리포트
    write_markdown_report(summary, pos_results, ccm)

    return summary


def write_markdown_report(
    summary: dict, pos_results: dict, ccm: set
) -> None:
    """사용자 읽기용 MD 리포트."""
    lines = []
    lines.append("# 우리말샘 vs ccomantle diff 결과")
    lines.append("")
    lines.append(f"_생성: {summary['generated_at']}_")
    lines.append(f"_source: {summary['source']}_")
    lines.append("")

    # 1. 처리 통계
    lines.append("## 1. 처리 통계")
    lines.append("")
    s = summary["processing_stats"]
    lines.append(f"- 우리말샘 총 entry: {s['total_entries']:,}")
    lines.append(f"- 일반어 아님 (방언/옛말/북한어) 제외: {s['filtered_non_general']:,}")
    lines.append(f"- 한글/길이 필터 제외: {s['filtered_invalid_word']:,}")
    lines.append(f"- 유지 (일반어 + 한글 1-6글자): {s['kept']:,}")
    lines.append(f"- ccomantle 사전 크기: {summary['ccomantle_size']:,}")
    lines.append("")

    # 2. POS 별 요약
    lines.append("## 2. POS 별 단어 수 (유니크) + ccomantle coverage")
    lines.append("")
    lines.append("| POS | 우리말샘 (유니크) | ccomantle 교집합 | 누락 | Coverage |")
    lines.append("|---|---:|---:|---:|---:|")
    for pos_key, label in [
        ("nouns", "명사"),
        ("verbs", "동사"),
        ("adjectives", "형용사"),
        ("adverbs", "부사"),
        ("others", "기타"),
    ]:
        r = pos_results[pos_key]
        lines.append(
            f"| {label} | {r['total']:,} | {r['overlap']:,} | "
            f"{r['missing']:,} | {r['coverage']:.2%} |"
        )
    lines.append("")

    # 3. 누락 명사 top
    lines.append("## 3. 누락 명사 — 사용자 통증 직격 (top 50)")
    lines.append("")
    lines.append("### 3.1 mecab 에도 있음 (전통 일반 명사, cost 정렬 낮은 순 = 빈도 ↑)")
    lines.append("")
    in_mecab = [m for m in pos_results["nouns"]["missing_list"] if m["source"] == "urimalsaem+mecab"]
    for i, m in enumerate(in_mecab[:50], 1):
        lines.append(f"{i:>2}. **{m['word']}** (cost={m['cost']})")
    lines.append("")

    lines.append("### 3.2 우리말샘 only (신어/전문용어/특수, alphabetical)")
    lines.append("")
    urim_only = [m for m in pos_results["nouns"]["missing_list"] if m["source"] == "urimalsaem_only"]
    for i, m in enumerate(urim_only[:50], 1):
        lines.append(f"{i:>2}. {m['word']}")
    lines.append("")

    # 4. 누락 동사·형용사 top
    for pos_key, label in [("verbs", "동사"), ("adjectives", "형용사")]:
        missing = pos_results[pos_key]["missing_list"]
        if not missing:
            continue
        lines.append(f"## 4. 누락 {label} (POS 정책 검토용, top 30)")
        lines.append("")
        for i, m in enumerate(missing[:30], 1):
            lines.append(f"{i:>2}. {m['word']}")
        lines.append("")

    # 5. 결정 필요 사항
    lines.append("## 5. 결정 필요 사항")
    lines.append("")
    lines.append("### 5.1 명사 추가 범위")
    n_nouns = pos_results["nouns"]["missing"]
    n_in_mecab = len(in_mecab)
    lines.append(f"- 누락 명사 총 {n_nouns:,} 개 (mecab 양쪽 {n_in_mecab:,} / 우리말샘 only {len(urim_only):,})")
    lines.append("- 보수: mecab 양쪽 top 5,000 만 추가 (안전, 전통 어휘)")
    lines.append("- 적극: 양쪽 전체 + 우리말샘 only top N (신어/외래어 포함)")
    lines.append("")
    lines.append("### 5.2 동사·형용사 정책")
    lines.append("- (가) 명사 only — 현재 정책 유지 (의미 정합성 ↑)")
    lines.append("- (나) 명사 + 기본형 동사·형용사 추가 (자유도 ↑)")
    lines.append("- (다) 활용형 클러스터링 — 큰 작업 (다음 PR)")
    lines.append("")
    lines.append("### 5.3 source 합집합 검토")
    lines.append("- 우리말샘 + mecab + ko 위키 (전 PR `feat/dict-diff-kowiki-titles` 산출)")
    lines.append("- 합집합으로 reference 강력화 가능. 단 ko 위키 only 는 일본 지명·인명 다수라 noise")
    lines.append("")

    md_path = DATA_DIR / "quality_report_urimalsaem.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="우리말샘 vs ccomantle diff")
    parser.add_argument("--no-mecab-crossref", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("ccomantle vs 우리말샘 diff")
    print("=" * 60)

    print("\n[1/4] ccomantle 사전 로드")
    ccm = load_ccomantle_dict()
    print(f"   ccomantle: {len(ccm):,} 단어")

    print("\n[2/4] mecab NNG cost cross-ref" + (" skip" if args.no_mecab_crossref else ""))
    mecab_costs = {} if args.no_mecab_crossref else load_mecab_nng_costs()
    if mecab_costs:
        print(f"   mecab NNG: {len(mecab_costs):,} 단어")

    print("\n[3/4] 우리말샘 25 chunk 처리 (~1.2M entry, 1-2분 소요)")
    pos_sets, stats = process_chunks()
    print(f"   처리 통계:")
    print(f"     총 entry: {stats['total_entries']:,}")
    print(f"     일반어 아님 제외: {stats['filtered_non_general']:,}")
    print(f"     한글/길이 제외: {stats['filtered_invalid_word']:,}")
    print(f"     유지: {stats['kept']:,}")
    print(f"   POS 별 유니크 단어:")
    for k, v in pos_sets.items():
        print(f"     {k}: {len(v):,}")

    print("\n[4/4] diff + 파일 출력")
    summary = write_outputs(pos_sets, stats, ccm, mecab_costs)

    # 터미널 요약 표
    print("\n" + "=" * 60)
    print("POS 별 coverage 요약 (상세는 MD 리포트)")
    print("=" * 60)
    print(f"{'POS':<12}{'total':>10}{'overlap':>10}{'missing':>10}{'coverage':>12}")
    for pos_key in ["nouns", "verbs", "adjectives", "adverbs", "others"]:
        r = summary["pos_results"][pos_key]
        print(
            f"{pos_key:<12}{r['total']:>10,}{r['overlap']:>10,}"
            f"{r['missing']:>10,}{r['coverage']:>11.2%}"
        )

    print("\n" + "=" * 60)
    print("저장 완료 — 다음은 파일 열어서 확인")
    print("=" * 60)
    print(f"  📄 data/quality_report_urimalsaem.md       ← 사용자 읽기용 ★")
    print(f"  📄 data/quality_report_urimalsaem.json    ← 머신 읽기용")
    print(f"  📄 data/missing_words_urimalsaem_nouns.json")
    print(f"  📄 data/missing_words_urimalsaem_verbs.json")
    print(f"  📄 data/missing_words_urimalsaem_adj.json")
    print(f"  📄 data/missing_words_urimalsaem_adverbs.json")
    print()
    print("다음 단계:")
    print("  1. MD 리포트 열어서 §2 POS 요약 + §3 명사 top 50 spot-check")
    print("  2. §5 결정 사항 확인 (명사 추가 범위 + POS 정책)")
    print("  3. 결정 후 expand_dict.py 확장해서 보강 진행")

    return 0


if __name__ == "__main__":
    sys.exit(main())
