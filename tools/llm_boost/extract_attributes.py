"""
tools/llm_boost/extract_attributes.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
무엇 (What)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
정답 단어에 대한 의미 속성·연관 단어를 Gemini Flash 로 추출 후 캐시 저장.
ccomantle 의 임베딩 cosine 한계 (사과↔빨강 < 사과↔파랑 같은 미세 fail) 를
LLM 의 상식 추론으로 보완. game.py 의 점수 계산 보정에 사용.

왜 (Why)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
임베딩 모델 (KoE5 등) 은 단어의 표면 동시발생 통계로만 의미 학습 → 상식
("사과는 빨갛다") 같은 자유 연상 약함. LLM 은 reasoning 으로 그걸 잡음.
운영자가 정답 설정 시 1회 호출 + 캐싱 → 비용 거의 0 (Gemini Flash 무료
한도). 동일 정답 재사용 시 LLM 호출 X.

설계
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 한 정답 단어 → JSON 응답:
  {
    "attributes":    ["빨강", "달콤", "둥근", ...] (5-10),
    "related_words": ["과일", "사과나무", "씨", ...] (10-30)
  }
- 저장: data/answer_boost_cache/<word>.json
- game.py 는 이 cache 로드 → 임베딩 vector boost 결합 (별 PR 단계)

사용 (Usage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # .env 에 GEMINI_API_KEY 설정 (Google AI Studio 에서 무료 발급)
    # https://aistudio.google.com/apikey

    pip install google-generativeai

    # 단일 단어
    python tools/llm_boost/extract_attributes.py --word 사과

    # 강제 재추출 (캐시 무시)
    python tools/llm_boost/extract_attributes.py --word 사과 --force

    # 여러 단어 한 번에
    python tools/llm_boost/extract_attributes.py --words 사과,배,강아지
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "answer_boost_cache"

MODEL_NAME = "gemini-flash-lite-latest"  # alias: free tier 최신 lite 모델 자동. 2.0/1.5 specific version issue 회피

PROMPT_TEMPLATE = """다음 한국어 단어의 의미·속성·연관 단어를 추출해주세요.

단어: {word}

게임에서 이 단어가 정답일 때, 의미적으로 가까운 단어들을 식별하는 데
사용됩니다. 일반 한국어 사용자의 직관에 맞는 연상을 우선합니다.

다음 JSON 형식으로만 답하세요 (다른 텍스트 없이):

{{
  "definition": "단어의 핵심 의미 한 줄",
  "attributes": ["속성1", "속성2", ...],
  "related_words": ["연관 단어1", "연관 단어2", ...]
}}

지침:
- attributes: 색깔·맛·크기·계절·재료·용도·감정 등 핵심 속성 5-10개.
  **형용사·동사 속성은 기본형 + 활용형 같이 포함** (게임에서 사용자가 어느 형태로 입력해도 매칭):
    - "달콤" → "달콤", "달콤한", "달콤하다"
    - "둥근" → "둥글다", "둥근", "둥글기"
    - "아삭함" → "아삭한", "아삭하다"
  → 이런 경우 attributes 안에 3개 다 포함 (5-10개의 의미 단위 기준이라 다소 늘어나도 OK)
- related_words: 유의어·상위어·하위어·연상어·범주 10-30개. 명사 위주.
- 모두 한국어, 1-6글자.
- "사과" 면 "빨강"·"달콤"·"달콤한"·"과일"·"사과나무" 같은 결.
- 임베딩 사전과 매칭되도록 단일 단어 형태."""


def load_env() -> None:
    """python-dotenv 가 있으면 .env 로드 (옵션)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        pass


def cache_path(word: str) -> Path:
    return CACHE_DIR / f"{word}.json"


def load_cache(word: str) -> dict | None:
    p = cache_path(word)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_cache(word: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(word)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_one(word: str, force: bool = False) -> dict:
    """단어 1개 추출 (캐시 우선). force=True 면 캐시 무시 + 덮어쓰기."""
    if not force:
        cached = load_cache(word)
        if cached is not None:
            print(f"   📦 cache hit: {word}")
            return cached

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY 환경변수 없음.\n"
            "1) https://aistudio.google.com/apikey 에서 발급 (무료)\n"
            "2) .env 에 GEMINI_API_KEY=... 추가 또는 export"
        )

    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai 패키지 없음.\n"
            "  pip install google-generativeai\n"
            "또는 uv pip install google-generativeai"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = PROMPT_TEMPLATE.format(word=word)

    t0 = time.perf_counter()
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )
    elapsed = time.perf_counter() - t0

    # 응답 파싱
    raw = response.text
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM 응답 JSON 파싱 실패: {e}\n응답: {raw[:300]}")

    # 메타 추가
    parsed["_meta"] = {
        "word": word,
        "model": MODEL_NAME,
        "elapsed_sec": round(elapsed, 2),
    }

    # 저장
    save_cache(word, parsed)
    return parsed


def summarize(word: str, data: dict) -> None:
    attrs = data.get("attributes", [])
    rels = data.get("related_words", [])
    defn = data.get("definition", "")
    elapsed = data.get("_meta", {}).get("elapsed_sec")

    print(f"\n=== \"{word}\" ===")
    if defn:
        print(f"   정의: {defn}")
    print(f"   속성 ({len(attrs)}): {', '.join(attrs)}")
    print(f"   연관 ({len(rels)}): {', '.join(rels[:20])}")
    if len(rels) > 20:
        print(f"           ... + {len(rels) - 20} more")
    if elapsed:
        print(f"   ⏱  {elapsed}s")


def main() -> int:
    load_env()

    parser = argparse.ArgumentParser(description="정답 단어 LLM 속성 추출 + 캐싱")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--word", help="단일 단어")
    g.add_argument("--words", help="여러 단어 (콤마로 구분, 예: 사과,배,강아지)")
    parser.add_argument("--force", action="store_true", help="캐시 무시 + 재추출")
    args = parser.parse_args()

    words = [w.strip() for w in (args.words or args.word).split(",") if w.strip()]

    print(f"단어 {len(words)}개 처리 (model={MODEL_NAME})\n")

    for w in words:
        try:
            data = extract_one(w, force=args.force)
            summarize(w, data)
        except Exception as e:
            print(f"\n❌ \"{w}\" 실패: {e}", file=sys.stderr)

    print(f"\n저장 위치: {CACHE_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
