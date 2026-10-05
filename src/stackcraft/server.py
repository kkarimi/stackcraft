"""Local authoritative game API and packaged browser assets."""

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from stackcraft.engine import legal_actions, new_game, step
from stackcraft.replay import make_replay, replay_states
from stackcraft.schema import GameState

MAX_SESSIONS = 128
MAX_REPLAY_ACTIONS = 2000
WEB = Path(__file__).parent / "web"


class NewGame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    seed: Annotated[int, Field(ge=0, le=2**53 - 1)] = 42


class Move(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action_id: Annotated[str, Field(min_length=1, max_length=16)]
    expected_pieces: Annotated[int, Field(ge=0)]


@dataclass
class Session:
    state: GameState
    actions: list[str] = field(default_factory=list)


def snapshot(state: GameState, session_id: str | None = None) -> dict[str, Any]:
    """Expose visible state only. Seeds and future pieces stay out of observations."""
    result = {
        "board": state.board,
        "current": state.current,
        "next_piece": state.next_piece,
        "score": state.score,
        "lines": state.lines,
        "pieces": state.piece_index,
        "terminal": state.terminal,
        "legal_actions": [asdict(action) for action in legal_actions(state)],
    }
    if session_id is not None:
        result["id"] = session_id
    return result


def create_app() -> FastAPI:
    app = FastAPI(title="Stackcraft", version="0.1.0")
    sessions: OrderedDict[str, Session] = OrderedDict()
    lock = RLock()

    def get_session(session_id: str) -> Session:
        if session_id not in sessions:
            raise HTTPException(404, "Game not found or expired. Start a new game.")
        sessions.move_to_end(session_id)
        return sessions[session_id]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/games")
    def create_game(request: NewGame) -> dict[str, Any]:
        with lock:
            session_id = uuid4().hex
            sessions[session_id] = Session(new_game(request.seed))
            if len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
            return snapshot(sessions[session_id].state, session_id)

    @app.get("/api/games/{session_id}")
    def read_game(session_id: str) -> dict[str, Any]:
        with lock:
            return snapshot(get_session(session_id).state, session_id)

    @app.post("/api/games/{session_id}/moves")
    def move(session_id: str, request: Move) -> dict[str, Any]:
        with lock:
            session = get_session(session_id)
            if request.expected_pieces != session.state.piece_index:
                raise HTTPException(409, "Game changed. Refresh the board before placing again.")
            if len(session.actions) >= MAX_REPLAY_ACTIONS:
                raise HTTPException(422, "This local session reached its 2,000-piece limit.")
            try:
                transition = step(session.state, request.action_id)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            session.state = transition.state
            session.actions.append(request.action_id)
            return snapshot(session.state, session_id)

    @app.get("/api/games/{session_id}/replay")
    def download_replay(session_id: str) -> dict[str, Any]:
        with lock:
            session = get_session(session_id)
            return make_replay(session.state.seed, session.actions)

    @app.post("/api/replays")
    def load_replay(artifact: dict[str, Any]) -> dict[str, Any]:
        actions = artifact.get("actions")
        if not isinstance(actions, list) or len(actions) > MAX_REPLAY_ACTIONS:
            raise HTTPException(422, "Replay requires an actions list of at most 2,000 moves.")
        try:
            states = replay_states(artifact)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"frames": [snapshot(state) for state in states]}

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app
