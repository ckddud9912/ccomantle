from typing import Optional

from pydantic import BaseModel


class SetAnswerRequest(BaseModel):
    answer: str


class GuessRequest(BaseModel):
    team: str
    word: str
    team_color: Optional[str] = None


class RoundRequest(BaseModel):
    round: int
