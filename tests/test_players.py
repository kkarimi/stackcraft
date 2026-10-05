from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from stackcraft.engine import new_game
from stackcraft.players import (
    BoardFeatures,
    Decision,
    HeuristicPlayer,
    RandomPlayer,
    board_features,
    observe,
    validate_decision,
)


def test_observation_has_no_hidden_future_or_seed() -> None:
    state = new_game(12)
    observation = observe(state)
    assert set(asdict(observation)) == {
        "board",
        "current",
        "next_piece",
        "legal_actions",
        "rules_version",
    }
    assert observe(replace(state, seed=555, piece_index=77)) == observation
    with pytest.raises(FrozenInstanceError):
        observation.current = "I"  # ty: ignore[invalid-assignment]


def test_player_choices_do_not_depend_on_hidden_seed() -> None:
    state = new_game(9)
    other = replace(state, seed=271, piece_index=9)
    assert HeuristicPlayer().choose(observe(state)) == HeuristicPlayer().choose(observe(other))
    assert RandomPlayer(81).choose(observe(state)) == RandomPlayer(81).choose(observe(other))


def test_random_has_independent_reproducible_rng() -> None:
    import random

    before = random.getstate()
    observation = observe(new_game(7))
    first, second = RandomPlayer(33), RandomPlayer(33)
    for _ in range(12):
        decision = first.choose(observation)
        assert decision == second.choose(observation)
        validate_decision(decision, observation)
    assert random.getstate() == before


def test_features_count_buried_holes_and_adjacent_height_differences() -> None:
    rows = [[0] * 10 for _ in range(20)]
    rows[17][0] = rows[19][0] = rows[19][1] = 1
    board = tuple(tuple(row) for row in rows)
    assert board_features(board) == BoardFeatures(4, 1, 3)


def test_heuristic_clears_line_instead_of_building_high_stack() -> None:
    state = new_game(0)
    board = state.board[:-1] + ((0, 0, 0, 0, 1, 1, 1, 1, 1, 1),)
    observation = observe(replace(state, board=board, current="I"))
    decision = HeuristicPlayer().choose(observation)
    assert decision.action_id == "r0x0"
    assert decision.probabilities is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.1"])
def test_probability_validation_rejects_invalid_values(bad: object) -> None:
    observation = observe(new_game(0))
    probabilities = {
        action.id: 1 / len(observation.legal_actions) for action in observation.legal_actions
    }
    probabilities[observation.legal_actions[0].id] = bad  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError, match="finite numbers"):
        validate_decision(Decision(observation.legal_actions[0].id, probabilities), observation)


def test_decision_validation_rejects_illegal_ids_and_incomplete_distributions() -> None:
    observation = observe(new_game(0))
    action_id = observation.legal_actions[0].id
    with pytest.raises(ValueError, match="illegal action"):
        validate_decision(Decision("not-legal"), observation)
    with pytest.raises(ValueError, match="cover every"):
        validate_decision(Decision(action_id, {action_id: 1.0}), observation)
    with pytest.raises(ValueError, match="sum to one"):
        validate_decision(
            Decision(action_id, {a.id: 0.0 for a in observation.legal_actions}), observation
        )
    with pytest.raises(ValueError, match="return a Decision"):
        validate_decision(None, observation)  # ty: ignore[invalid-argument-type]
