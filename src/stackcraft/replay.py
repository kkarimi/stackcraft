"""Portable action replays, checked by running the authoritative engine again."""

from typing import Any

from stackcraft.engine import new_game, step
from stackcraft.schema import RULES_VERSION, GameState

SCHEMA_VERSION = 1


def _summary(state: GameState) -> dict[str, int | bool]:
    return {
        "score": state.score,
        "lines": state.lines,
        "pieces": state.piece_index,
        "terminal": state.terminal,
    }


def make_replay(seed: int, actions: list[str]) -> dict[str, Any]:
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "rules_version": RULES_VERSION,
        "seed": seed,
        "actions": list(actions),
    }
    frames = replay_states(artifact)
    artifact["final"] = _summary(frames[-1])
    return artifact


def replay_states(artifact: dict[str, Any]) -> list[GameState]:
    """Reject malformed actions, incompatible versions, and false summaries."""
    if not isinstance(artifact, dict):
        raise ValueError("replay must be a JSON object")
    version = artifact.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError("unsupported replay schema_version")
    if artifact.get("rules_version") != RULES_VERSION:
        raise ValueError("unsupported replay rules_version")
    seed = artifact.get("seed")
    if type(seed) is not int:
        raise ValueError("replay seed must be an integer")
    actions = artifact.get("actions")
    if not isinstance(actions, list) or any(not isinstance(a, str) for a in actions):
        raise ValueError("replay actions must be a list of action IDs")
    states = [new_game(seed)]
    for index, action in enumerate(actions):
        try:
            states.append(step(states[-1], action).state)
        except ValueError as error:
            raise ValueError(f"invalid replay move {index}: {error}") from error
    if "final" in artifact:
        expected = _summary(states[-1])
        final = artifact["final"]
        if (
            not isinstance(final, dict)
            or final.keys() != expected.keys()
            or any(type(final[key]) is not type(value) for key, value in expected.items())
            or final != expected
        ):
            raise ValueError("replay final summary does not match simulated outcome")
    return states
