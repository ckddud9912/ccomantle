import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.embeddings import EmbeddingStore


MAX_ROUNDS = 6
TARGET_TOP1000 = 0.63  # 1000위 목표 유사도


def _scale_positive(arr: np.ndarray, alpha: float) -> np.ndarray:
    """음수는 0으로, 양수는 x ** alpha 로 변환."""
    return np.clip(arr, 0.0, None) ** alpha


def _scale_scalar(x: float, alpha: float) -> float:
    if x <= 0:
        return 0.0
    return x ** alpha


@dataclass
class GameState:
    store: EmbeddingStore

    answer_word: Optional[str] = None
    answer_vector: Optional[np.ndarray] = None
    word_to_rank: Dict[str, int] = field(default_factory=dict)

    sim_alpha: float = 1.0
    sim_top1: Optional[float] = None
    sim_top20: Optional[float] = None
    sim_top1000: Optional[float] = None

    current_round: int = 1
    rounds: Dict[int, List[Dict]] = field(
        default_factory=lambda: {i: [] for i in range(1, MAX_ROUNDS + 1)}
    )
    finished: bool = False
    team_colors: Dict[str, str] = field(default_factory=dict)

    # ----- 정답 설정 -----
    def reset_for_answer(self, answer: str) -> None:
        if answer not in self.store:
            raise KeyError(answer)

        self.answer_word = answer
        self.answer_vector = self.store.vector(answer)
        self.rounds = {i: [] for i in range(1, MAX_ROUNDS + 1)}
        self.current_round = 1
        self.finished = False
        self.team_colors = {}

        self._compute_rankings()

    def _compute_rankings(self) -> None:
        sims = self.store.matrix @ self.answer_vector  # (N,)
        order = np.argsort(-sims)

        self.word_to_rank = {
            self.store.words[idx]: rank + 1 for rank, idx in enumerate(order)
        }

        n = len(sims)
        top1 = float(sims[order[0]]) if n > 0 else 0.0
        top20 = float(sims[order[min(19, n - 1)]]) if n > 0 else 0.0
        top1000 = float(sims[order[min(999, n - 1)]]) if n > 0 else 0.0

        if 0 < top1000 < 1:
            self.sim_alpha = math.log(TARGET_TOP1000) / math.log(top1000)
        else:
            self.sim_alpha = 1.0

        print(f"[INFO] sim_top1000_raw={top1000:.4f}, SIM_ALPHA={self.sim_alpha:.4f}")

        self.sim_top1 = _scale_scalar(top1, self.sim_alpha)
        self.sim_top20 = _scale_scalar(top20, self.sim_alpha)
        self.sim_top1000 = _scale_scalar(top1000, self.sim_alpha)

    # ----- 라운드 -----
    def set_round(self, r: int) -> None:
        if not (1 <= r <= MAX_ROUNDS):
            raise ValueError(f"invalid round: {r}")
        self.current_round = r

    # ----- 추측 -----
    def submit_guess(self, team: str, word: str, color: str) -> Dict:
        if self.finished:
            return {"result": "error", "error": "경기가 종료되었습니다."}

        if self.answer_word is None:
            return {"result": "error", "error": "정답이 아직 설정되지 않았습니다."}

        if team not in self.team_colors:
            self.team_colors[team] = color

        for s in self.rounds[self.current_round]:
            if s["team"] == team:
                return {"result": "duplicate"}

        if word == self.answer_word:
            entry = self._make_entry(team, word, is_answer=True, rank=1, similarity=1.0)
            self.rounds[self.current_round].append(entry)
            return {"result": "correct", "entry": entry}

        if word not in self.store:
            return {"result": "error", "error": "사전에 없는 단어입니다."}

        raw = float(self.answer_vector @ self.store.vector(word))
        sim = round(_scale_scalar(raw, self.sim_alpha), 3)
        rank = self.word_to_rank.get(word)

        entry = self._make_entry(team, word, is_answer=False, rank=rank, similarity=sim)
        self.rounds[self.current_round].append(entry)
        return {"result": "ok", "entry": entry}

    def _make_entry(self, team: str, word: str, *, is_answer: bool, rank, similarity: float) -> Dict:
        return {
            "round": self.current_round,
            "team": team,
            "team_color": self.team_colors[team],
            "word": word,
            "is_answer": is_answer,
            "rank": rank,
            "similarity": similarity,
        }

    # ----- 조회 -----
    def leaderboard(self) -> Dict:
        sorted_rounds = {
            str(r): sorted(self.rounds[r], key=lambda x: x["similarity"], reverse=True)
            for r in self.rounds
        }
        return {
            "current_round": self.current_round,
            "max_rounds": MAX_ROUNDS,
            "answer": self.answer_word,
            "sim_top1": self.sim_top1,
            "sim_top20": self.sim_top20,
            "sim_top1000": self.sim_top1000,
            "rounds": sorted_rounds,
            "finished": self.finished,
        }

    def top1000(self) -> List[Dict]:
        sims_raw = self.store.matrix @ self.answer_vector
        sims = _scale_positive(sims_raw, self.sim_alpha)
        order = np.argsort(-sims)[:1000]
        return [
            {"rank": i + 1, "word": self.store.words[idx], "similarity": round(float(sims[idx]), 4)}
            for i, idx in enumerate(order)
        ]

    # ----- 종료 / 최종 성적 -----
    def end(self) -> None:
        self.finished = True

    def final_result(self) -> List[Dict]:
        scores: Dict[str, List[float]] = {}
        for r in self.rounds.values():
            for s in r:
                if isinstance(s["similarity"], float):
                    scores.setdefault(s["team"], []).append(s["similarity"])

        result = [
            {
                "team": t,
                "team_color": self.team_colors.get(t),
                "avg": round(sum(sims) / len(sims), 4),
            }
            for t, sims in scores.items()
        ]
        result.sort(key=lambda x: x["avg"], reverse=True)
        return result
