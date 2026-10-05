"""Paired, replayable episodes with explicit failures and decision timings."""

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from stackcraft.engine import new_game, step
from stackcraft.players import Player, observe, validate_decision
from stackcraft.replay import make_replay
from stackcraft.schema import RULES_VERSION


def _timing(seconds: list[float]) -> dict[str, float | int]:
    ordered = sorted(seconds)
    return {
        "count": len(seconds),
        "total_seconds": sum(seconds),
        "mean_seconds": statistics.mean(seconds) if seconds else 0.0,
        "median_seconds": statistics.median(seconds) if seconds else 0.0,
        "p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else 0.0,
    }


def run_episode(player: Player, seed: int, max_pieces: int = 100) -> dict[str, Any]:
    """Stop at top-out, cap, or first error; never replace a failed player move."""
    if type(max_pieces) is not int or max_pieces < 1:
        raise ValueError("max_pieces must be a positive integer")
    state = new_game(seed)
    actions: list[str] = []
    decisions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seconds: list[float] = []
    while not state.terminal and state.piece_index < max_pieces:
        observation = observe(state)
        event: dict[str, Any] = {"turn": state.piece_index, "observation": asdict(observation)}
        started = time.perf_counter()
        try:
            decision = player.choose(observation)
        except Exception as error:
            duration = time.perf_counter() - started
            failure = {"turn": state.piece_index, "kind": "player_error", "message": str(error)}
            errors.append(failure)
            event["error"] = failure
        else:
            duration = time.perf_counter() - started
            try:
                validate_decision(decision, observation)
            except ValueError as error:
                failure = {
                    "turn": state.piece_index,
                    "kind": "invalid_decision",
                    "message": str(error),
                }
                errors.append(failure)
                event["error"] = failure
            else:
                event["action_id"] = decision.action_id
                event["probabilities"] = decision.probabilities
                state = step(state, decision.action_id).state
                actions.append(decision.action_id)
        event["decision_seconds"] = duration
        seconds.append(duration)
        decisions.append(event)
        if errors:
            break
    cap_hit = not state.terminal and not errors and state.piece_index == max_pieces
    return {
        "schema_version": 1,
        "rules_version": RULES_VERSION,
        "player": {"name": player.name, "revision": player.revision},
        "seed": seed,
        "max_pieces": max_pieces,
        "outcome": {
            "score": state.score,
            "lines": state.lines,
            "pieces": state.piece_index,
            "terminal": state.terminal,
            "cap_hit": cap_hit,
            "status": "error" if errors else "top_out" if state.terminal else "cap",
        },
        "errors": errors,
        "decisions": decisions,
        "timing": _timing(seconds),
        "replay": make_replay(seed, actions),
    }


def run_tournament(
    player_factories: Mapping[str, Callable[[], Player]],
    seeds: Sequence[int],
    max_pieces: int = 100,
) -> dict[str, Any]:
    """Fresh player per paired seed. Timings are not deterministic game content."""
    seeds = tuple(seeds)
    if not player_factories or not seeds:
        raise ValueError("tournament requires players and seeds")
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("tournament seeds must be distinct integers")
    episodes = []
    summaries = {}
    for label, factory in player_factories.items():
        group = [run_episode(factory(), seed, max_pieces) for seed in seeds]
        for episode in group:
            episode["player_id"] = label
        episodes.extend(group)
        summaries[label] = {
            "episodes": len(group),
            "mean_lines": statistics.mean(e["outcome"]["lines"] for e in group),
            "median_lines": statistics.median(e["outcome"]["lines"] for e in group),
            "mean_score": statistics.mean(e["outcome"]["score"] for e in group),
            "mean_pieces": statistics.mean(e["outcome"]["pieces"] for e in group),
            "cap_hit_rate": statistics.mean(e["outcome"]["cap_hit"] for e in group),
            "error_rate": statistics.mean(bool(e["errors"]) for e in group),
            "timing": _timing([d["decision_seconds"] for e in group for d in e["decisions"]]),
        }
    return {
        "schema_version": 1,
        "rules_version": RULES_VERSION,
        "seeds": list(seeds),
        "max_pieces": max_pieces,
        "summary": summaries,
        "episodes": episodes,
    }
