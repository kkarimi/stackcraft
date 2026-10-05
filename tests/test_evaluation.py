import copy
import math
from dataclasses import asdict

import pytest

from stackcraft.engine import new_game
from stackcraft.evaluation import paired_bootstrap, paired_report, summarize_positions
from stackcraft.players import Decision, observe
from stackcraft.schema import RULES_VERSION


def episode(player: str, seed: int, lines: int, *, failed: bool = False) -> dict:
    """Statistical fixture, not a claimed simulator run."""
    errors = [{"turn": 10, "kind": "player_error", "message": "toy failure"}] if failed else []
    return {
        "schema_version": 1,
        "rules_version": RULES_VERSION,
        "player_id": player,
        "player": {"name": player, "revision": "toy-v1", "runtime_config": {}},
        "seed": seed,
        "max_pieces": 100,
        "outcome": {
            "score": 100 * lines,
            "lines": lines,
            "pieces": 10,
            "terminal": not failed,
            "cap_hit": False,
            "status": "error" if failed else "top_out",
        },
        "errors": errors,
        "decisions": [{"decision_seconds": 0.01, "input_tokens": 100}] * (10 + failed),
    }


def report(episodes, **kwargs):
    return paired_report(
        episodes, trained_id="trained", base_id="base", bootstrap_samples=200, **kwargs
    )


def test_paired_constant_improvement_has_exact_interval() -> None:
    result = paired_bootstrap([3, 3, 3], samples=200)
    assert result["mean_difference"] == result["ci95_lower"] == result["ci95_upper"] == 3
    assert result == paired_bootstrap([3, 3, 3], samples=200)


def test_pairing_preserves_difficulty_and_is_order_independent() -> None:
    episodes = [
        episode("base", 80, 30),
        episode("trained", 80, 33),
        episode("base", 70, 1),
        episode("trained", 70, 4),
    ]
    result = report(episodes)
    assert result == report(episodes[::-1])
    assert result["seeds"] == [70, 80]
    assert result["paired_trained_minus_base"]["lines"]["ci95_lower"] == 3
    assert result["positive_mean_lines_signal"]
    assert not result["selection_performed"]
    assert not result["matches_reserved_test_pool"]
    assert result["players"]["trained"]["latency_seconds"]["p95"] == 0.01
    assert result["players"]["trained"]["input_tokens"]["mean"] == 100


def test_failure_is_counted_as_zero_and_partial_outcome_is_retained() -> None:
    result = report([episode("base", 70, 2), episode("trained", 70, 50, failed=True)])
    assert result["players"]["trained"]["episodes"] == 1
    assert result["players"]["trained"]["failure_adjusted"]["lines"]["mean"] == 0
    assert result["players"]["trained"]["observed_before_error"]["lines"]["mean"] == 50
    assert result["players"]["trained"]["error_rate"] == 1
    assert result["players"]["trained"]["failed_seeds"] == [70]
    assert result["paired_trained_minus_base"]["lines"]["mean_difference"] == -2
    assert not result["positive_mean_lines_signal"]


def test_positive_interval_cannot_hide_base_failures() -> None:
    rows = [
        episode(player, seed, 2, failed=player == "base")
        for player in ("base", "trained")
        for seed in (70, 80)
    ]
    result = report(rows)
    assert result["paired_trained_minus_base"]["lines"]["ci95_lower"] == 2
    assert not result["errors_acceptable"]
    assert not result["positive_mean_lines_signal"]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "cap",
        "rules",
        "revision",
        "runtime",
        "bad_metric",
        "bad_duration",
        "status",
    ],
)
def test_mismatched_or_invalid_episodes_are_rejected(mutation: str) -> None:
    rows = [episode(player, seed, 2) for player in ("base", "trained") for seed in (70, 80)]
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "cap":
        rows[-1]["max_pieces"] = 200
    elif mutation == "rules":
        rows[-1]["rules_version"] = "other"
    elif mutation == "revision":
        rows[-1]["player"]["revision"] = "later-checkpoint"
    elif mutation == "runtime":
        rows[-1]["player"]["runtime_config"] = {"max_length": 12}
    elif mutation == "bad_metric":
        rows[-1]["outcome"]["lines"] = float("nan")
    elif mutation == "bad_duration":
        rows[-1]["decisions"][0]["decision_seconds"] = -1
    else:
        rows[-1]["outcome"]["cap_hit"] = True
    with pytest.raises(ValueError):
        report(rows)


def positions():
    observation = observe(new_game(70))
    target = observation.legal_actions[0].id
    records = [
        {"id": str(index), "action_id": target, "observation": asdict(observation)}
        for index in range(2)
    ]
    probabilities = {a.id: 1 / len(observation.legal_actions) for a in observation.legal_actions}
    return records, target, probabilities


def test_position_metrics_are_separate_and_probability_scores_correct() -> None:
    records, target, probabilities = positions()
    result = summarize_positions(
        records, {row["id"]: Decision(target, probabilities) for row in records}
    )
    assert result["teacher_agreement"] == 1
    assert result["mean_nll"] == pytest.approx(math.log(len(probabilities)))
    assert result["mean_brier"] == pytest.approx(1 - 1 / len(probabilities))
    assert result["complete_probability_coverage"]
    assert not result["selection_performed"]


def test_missing_predictions_remain_in_agreement_denominator() -> None:
    records, target, probabilities = positions()
    result = summarize_positions(records, {"0": Decision(target, probabilities)})
    assert result["teacher_agreement"] == 0.5
    assert result["error_rate"] == 0.5
    assert result["failed_predictions"] == ["1"]
    assert result["mean_nll"] is None and result["mean_brier"] is None
    assert result["finite_subset_nll_positions"] == 1


def test_missing_probabilities_and_zero_probability_are_explicit() -> None:
    records, target, probabilities = positions()
    result = summarize_positions(records, {row["id"]: Decision(target) for row in records})
    assert result["teacher_agreement"] == 1
    assert result["mean_nll"] is None
    assert result["missing_probabilities"] == ["0", "1"]
    for action in probabilities:
        probabilities[action] = 0 if action == target else 1 / (len(probabilities) - 1)
    result = summarize_positions(
        records, {row["id"]: Decision(target, probabilities) for row in records}
    )
    assert result["nll_is_infinite"]
    assert result["zero_target_probability_count"] == 2
    assert result["mean_nll"] is None


def test_degenerate_single_episode_cannot_claim_positive_signal() -> None:
    result = report([episode("base", 70, 1), episode("trained", 70, 2)])
    assert result["paired_trained_minus_base"]["lines"]["ci95_lower"] == 1
    assert not result["positive_mean_lines_signal"]
