"""Players receive visible state only; episode seeds stay in the runner."""

import math
import random
from dataclasses import dataclass
from typing import Protocol

from stackcraft.engine import legal_actions, place
from stackcraft.schema import HEIGHT, RULES_VERSION, WIDTH, Board, GameState, Placement


@dataclass(frozen=True)
class Observation:
    board: Board
    current: str
    next_piece: str
    legal_actions: tuple[Placement, ...]
    rules_version: str = RULES_VERSION


@dataclass(frozen=True)
class Decision:
    action_id: str
    probabilities: dict[str, float] | None = None


class Player(Protocol):
    name: str
    revision: str

    def choose(self, observation: Observation) -> Decision: ...


def observe(state: GameState) -> Observation:
    """Do not expose seed, piece index, or any hidden random-generator state."""
    return Observation(state.board, state.current, state.next_piece, legal_actions(state))


def validate_decision(decision: Decision, observation: Observation) -> None:
    if not isinstance(decision, Decision):
        raise ValueError("player must return a Decision")
    ids = {action.id for action in observation.legal_actions}
    if not isinstance(decision.action_id, str) or decision.action_id not in ids:
        raise ValueError("player selected an illegal action")
    probabilities = decision.probabilities
    if probabilities is None:
        return
    if not isinstance(probabilities, dict) or probabilities.keys() != ids:
        raise ValueError("probabilities must cover every legal action exactly")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
        for value in probabilities.values()
    ):
        raise ValueError("probabilities must be finite numbers between zero and one")
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to one")


class RandomPlayer:
    """Uniform legal placement with an RNG independent of the piece generator."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        if type(seed) is not int:
            raise ValueError("player RNG seed must be an integer")
        self.revision = f"random-v1-rng-{seed}"
        self._rng = random.Random(seed)

    def choose(self, observation: Observation) -> Decision:
        if not observation.legal_actions:
            raise ValueError("cannot choose with no legal actions")
        actions = observation.legal_actions
        return Decision(
            self._rng.choice(actions).id,
            {action.id: 1.0 / len(actions) for action in actions},
        )


@dataclass(frozen=True)
class BoardFeatures:
    aggregate_height: int
    holes: int
    bumpiness: int


def board_features(board: Board) -> BoardFeatures:
    heights = []
    holes = 0
    for x in range(WIDTH):
        first = next((y for y in range(HEIGHT) if board[y][x]), HEIGHT)
        heights.append(HEIGHT - first)
        holes += sum(board[y][x] == 0 for y in range(first, HEIGHT))
    return BoardFeatures(
        sum(heights), holes, sum(abs(a - b) for a, b in zip(heights, heights[1:], strict=False))
    )


def board_value(board: Board, cleared: int) -> int:
    """Fixed, untuned v1 weights: -height -4*holes -bumpiness +8*lines."""
    features = board_features(board)
    return -features.aggregate_height - 4 * features.holes - features.bumpiness + 8 * cleared


class HeuristicPlayer:
    """Greedy afterboard value; ties use the engine's rotation/column order."""

    name = "heuristic"
    revision = "heuristic-v1-height1-holes4-bump1-lines8"

    def choose(self, observation: Observation) -> Decision:
        if not observation.legal_actions:
            raise ValueError("cannot choose with no legal actions")
        action = max(
            observation.legal_actions,
            key=lambda action: board_value(*place(observation.board, observation.current, action)),
        )
        return Decision(action.id)
