"""Pure paired evaluation; never chooses checkpoints or runs game episodes."""

import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from stackcraft.data import canonical_json
from stackcraft.players import Decision, observe, validate_decision
from stackcraft.schema import RULES_VERSION, GameState

FINAL_TEST_SEEDS = tuple(range(30000, 30200))
METRICS = ("lines", "score", "pieces")


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap(
    differences: Sequence[float], *, samples: int = 10000, seed: int = 2026
) -> dict[str, Any]:
    """Percentile 95% interval, resampling complete paired episode differences."""
    if not differences or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in differences
    ):
        raise ValueError("paired differences must be nonempty finite numbers")
    if type(samples) is not int or samples < 1 or type(seed) is not int:
        raise ValueError("bootstrap samples must be positive and seed an integer")
    rng = random.Random(seed)
    count = len(differences)
    means = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples)
    ]
    return {
        "episodes": count,
        "mean_difference": statistics.mean(differences),
        "ci95_lower": _percentile(means, 0.025),
        "ci95_upper": _percentile(means, 0.975),
        "method": "paired episode percentile bootstrap",
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1],
    }


def _validate_episode(episode: dict[str, Any]) -> None:
    if episode.get("schema_version") != 1 or type(episode.get("schema_version")) is not int:
        raise ValueError("unsupported episode schema")
    if episode.get("rules_version") != RULES_VERSION:
        raise ValueError("unsupported episode rules")
    if not isinstance(episode.get("player_id"), str) or not episode["player_id"]:
        raise ValueError("each episode needs a nonempty player_id")
    if type(episode.get("seed")) is not int:
        raise ValueError("episode seed must be an integer")
    cap = episode.get("max_pieces")
    if type(cap) is not int or cap < 1:
        raise ValueError("episode cap must be a positive integer")
    player = episode.get("player")
    if not isinstance(player, dict) or any(
        not isinstance(player.get(key), str) or not player[key] for key in ("name", "revision")
    ):
        raise ValueError("player name and revision are required")
    if not isinstance(player.get("runtime_config", {}), dict):
        raise ValueError("runtime configuration must be an object")
    outcome = episode.get("outcome")
    if not isinstance(outcome, dict) or any(
        type(outcome.get(metric)) is not int or outcome[metric] < 0 for metric in METRICS
    ):
        raise ValueError("outcome metrics must be nonnegative integers")
    if outcome["pieces"] > cap:
        raise ValueError("outcome exceeds piece cap")
    if any(type(outcome.get(field)) is not bool for field in ("terminal", "cap_hit")):
        raise ValueError("terminal and cap_hit must be booleans")
    errors, decisions = episode.get("errors"), episode.get("decisions")
    if not isinstance(errors, list) or not isinstance(decisions, list):
        raise ValueError("episode errors and decisions must be lists")
    if len(errors) > 1 or any(
        not isinstance(error, dict) or error.get("kind") not in ("player_error", "invalid_decision")
        for error in errors
    ):
        raise ValueError("invalid episode error record")
    expected_status = "error" if errors else "top_out" if outcome["terminal"] else "cap"
    expected_cap = not errors and not outcome["terminal"] and outcome["pieces"] == cap
    if outcome.get("status") != expected_status or outcome["cap_hit"] != expected_cap:
        raise ValueError("outcome status is inconsistent with errors, terminal state, or cap")
    if not errors and not outcome["terminal"] and outcome["pieces"] != cap:
        raise ValueError("successful nonterminal episode ended before its cap")
    if errors and outcome["terminal"]:
        raise ValueError("an errored episode cannot also finish in top-out")
    if len(decisions) != outcome["pieces"] + bool(errors):
        raise ValueError("decision count does not match placements and errors")
    for event in decisions:
        duration = event.get("decision_seconds") if isinstance(event, dict) else None
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise ValueError("decision duration must be a finite nonnegative number")
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("decision duration must be a finite nonnegative number")
        tokens = event.get("input_tokens")
        if tokens is not None and (type(tokens) is not int or tokens < 1):
            raise ValueError("input_tokens must be a positive integer when provided")


def paired_report(
    episodes: Sequence[dict[str, Any]],
    *,
    trained_id: str,
    base_id: str,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 2026,
    max_error_rate: float = 0.0,
) -> dict[str, Any]:
    """Failure-adjusted primary outcomes keep failed episodes at zero, never drop them.

    Observed partial outcomes are also reported. A positive interval is statistical
    evidence for this comparison, not automatic checkpoint selection or publication.
    """
    if not episodes or trained_id == base_id:
        raise ValueError("report requires episodes and distinct trained/base players")
    if type(max_error_rate) not in (int, float) or not 0 <= max_error_rate <= 1:
        raise ValueError("max_error_rate must be between zero and one")
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    identities: dict[str, str] = {}
    caps = set()
    for episode in episodes:
        _validate_episode(episode)
        player_id, seed = episode["player_id"], episode["seed"]
        if seed in grouped[player_id]:
            raise ValueError("duplicate player/seed episode")
        grouped[player_id][seed] = episode
        caps.add(episode["max_pieces"])
        identity = canonical_json(episode["player"])
        if identities.setdefault(player_id, identity) != identity:
            raise ValueError("player revision or runtime changed within evaluation")
    if trained_id not in grouped or base_id not in grouped:
        raise ValueError("both requested players must be present")
    seeds = sorted(grouped[base_id])
    if len(caps) != 1 or any(sorted(group) != seeds for group in grouped.values()):
        raise ValueError("all players must have identical seed sets and episode caps")
    summary: dict[str, Any] = {}
    for player_id, group in grouped.items():
        ordered = [group[seed] for seed in seeds]
        failed = sum(bool(episode["errors"]) for episode in ordered)
        decisions = [event for episode in ordered for event in episode["decisions"]]
        primary = {
            metric: [0 if e["errors"] else e["outcome"][metric] for e in ordered]
            for metric in METRICS
        }
        summary[player_id] = {
            "identity": json.loads(identities[player_id]),
            "episodes": len(ordered),
            "failed_episodes": failed,
            "error_rate": failed / len(ordered),
            "invalid_decisions": sum(
                error["kind"] == "invalid_decision" for e in ordered for error in e["errors"]
            ),
            "cap_hit_rate": statistics.mean(e["outcome"]["cap_hit"] for e in ordered),
            "failure_adjusted": {
                metric: _numeric_summary(values) for metric, values in primary.items()
            },
            "observed_before_error": {
                metric: _numeric_summary([e["outcome"][metric] for e in ordered])
                for metric in METRICS
            },
            "latency_seconds": _numeric_summary([d["decision_seconds"] for d in decisions]),
            "input_tokens": _numeric_summary(
                [d["input_tokens"] for d in decisions if d.get("input_tokens") is not None]
            ),
            "failed_seeds": [e["seed"] for e in ordered if e["errors"]],
        }
    comparison = {}
    for metric in METRICS:
        differences = []
        for seed in seeds:
            trained, base = grouped[trained_id][seed], grouped[base_id][seed]
            difference = (0 if trained["errors"] else trained["outcome"][metric]) - (
                0 if base["errors"] else base["outcome"][metric]
            )
            differences.append(difference)
        comparison[metric] = paired_bootstrap(
            differences, samples=bootstrap_samples, seed=bootstrap_seed
        )
    errors_acceptable = all(
        summary[player]["error_rate"] <= max_error_rate for player in (trained_id, base_id)
    )
    return {
        "schema_version": 1,
        "rules_version": RULES_VERSION,
        "seeds": seeds,
        "matches_reserved_test_pool": tuple(seeds) == FINAL_TEST_SEEDS,
        "max_pieces": next(iter(caps)),
        "trained_id": trained_id,
        "base_id": base_id,
        "primary_failure_policy": "zero lines/score/pieces on errors; retain all pairs",
        "players": summary,
        "paired_trained_minus_base": comparison,
        "max_error_rate": max_error_rate,
        "errors_acceptable": errors_acceptable,
        "positive_mean_lines_signal": (
            len(seeds) >= 2 and comparison["lines"]["ci95_lower"] > 0 and errors_acceptable
        ),
        "selection_performed": False,
    }


def summarize_positions(
    records: Sequence[dict[str, Any]], predictions: Mapping[str, Decision | None]
) -> dict[str, Any]:
    """Teacher agreement and proper probability scores, separately from game outcomes.

    Missing/invalid predictions count against agreement. NLL/Brier require complete
    probabilities; if any prediction fails or omits them, aggregate scores are null.
    Finite-subset diagnostic scores are labeled with their smaller denominator.
    """
    ids = [row["id"] for row in records]
    if not ids or len(set(ids)) != len(ids) or set(predictions) - set(ids):
        raise ValueError("position IDs must be nonempty, unique, and match prediction keys")
    correct = 0
    failures = []
    missing_probabilities = []
    nll = []
    brier = []
    for row in records:
        raw = row["observation"]
        state = GameState(
            tuple(tuple(r) for r in raw["board"]), 0, 0, raw["current"], raw["next_piece"]
        )
        observation = observe(state)
        target = row["action_id"]
        if target not in {action.id for action in observation.legal_actions}:
            raise ValueError("teacher label is illegal")
        decision = predictions.get(row["id"])
        try:
            if decision is None:
                raise ValueError("missing prediction")
            validate_decision(decision, observation)
        except ValueError:
            failures.append(row["id"])
            continue
        correct += decision.action_id == target
        probabilities = decision.probabilities
        if probabilities is None:
            missing_probabilities.append(row["id"])
            continue
        # A stated zero target probability means infinite NLL; preserve that fact
        # explicitly without putting nonstandard Infinity values into JSON.
        nll.append(-math.log(probabilities[target]) if probabilities[target] > 0 else None)
        brier.append(sum((p - int(key == target)) ** 2 for key, p in probabilities.items()))
    complete = len(brier) == len(records)
    finite_nll = [value for value in nll if value is not None]
    zero_count = len(nll) - len(finite_nll)
    return {
        "positions": len(records),
        "correct": correct,
        "teacher_agreement": correct / len(records),
        "failed_predictions": failures,
        "error_rate": len(failures) / len(records),
        "missing_probabilities": missing_probabilities,
        "probability_positions": len(brier),
        "complete_probability_coverage": complete,
        "mean_nll": statistics.mean(finite_nll) if complete and not zero_count else None,
        "nll_is_infinite": bool(zero_count),
        "zero_target_probability_count": zero_count,
        "mean_brier": statistics.mean(brier) if complete else None,
        "finite_subset_mean_nll": statistics.mean(finite_nll) if finite_nll else None,
        "finite_subset_nll_positions": len(finite_nll),
        "available_subset_mean_brier": statistics.mean(brier) if brier else None,
        "selection_performed": False,
    }
