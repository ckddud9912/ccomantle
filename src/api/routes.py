from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from api.schemas import GuessRequest, RoundRequest, SetAnswerRequest
from core.game import GameState


router = APIRouter()


def get_game(request: Request) -> GameState:
    return request.app.state.game


@router.post("/set_answer")
async def set_answer(req: SetAnswerRequest, game: GameState = Depends(get_game)):
    try:
        game.reset_for_answer(req.answer)
    except KeyError:
        return JSONResponse({"error": "사전에 없는 단어입니다."})
    return {"status": "ok", "answer": game.answer_word}


@router.post("/set_round")
async def set_round(req: RoundRequest, game: GameState = Depends(get_game)):
    try:
        game.set_round(req.round)
    except ValueError:
        return JSONResponse({"error": "Invalid round"}, status_code=400)
    return {"status": "ok", "current_round": game.current_round}


@router.post("/guess")
async def guess(req: GuessRequest, game: GameState = Depends(get_game)):
    team = req.team.strip()
    word = req.word.strip()
    color = (req.team_color or "#3b82f6").strip()

    result = game.submit_guess(team, word, color)
    if result.get("result") == "error":
        return JSONResponse({"error": result["error"]})
    return result


@router.get("/leaderboard")
async def leaderboard(game: GameState = Depends(get_game)):
    return game.leaderboard()


@router.get("/top1000")
async def top1000(game: GameState = Depends(get_game)):
    if game.answer_word is None:
        return JSONResponse({"error": "정답이 아직 설정되지 않았습니다."})
    return {"answer": game.answer_word, "top1000": game.top1000()}


@router.post("/end_game")
async def end_game(game: GameState = Depends(get_game)):
    game.end()
    return {"status": "finished"}


@router.get("/final_result")
async def final_result(game: GameState = Depends(get_game)):
    return {"result": game.final_result()}
