from dataclasses import replace

import pytest

from stackcraft import engine
from stackcraft.engine import new_game, place
from stackcraft.expert import SearchExpert
from stackcraft.players import observe


def test_expert_never_queries_hidden_sequence(monkeypatch) -> None:
    observation = observe(new_game(43))

    def forbidden(*args, **kwargs):
        raise AssertionError("expert queried hidden future")

    monkeypatch.setattr(engine, "piece_at", forbidden)
    monkeypatch.setattr(engine, "step", forbidden)
    expert = SearchExpert()
    values = expert.action_values(observation)
    assert set(values) == {action.id for action in observation.legal_actions}
    assert expert.choose(observation).action_id == max(values, key=lambda a: values[a])


def test_search_is_independent_of_real_seed_and_turn() -> None:
    state = new_game(3)
    alternate = replace(state, seed=654123, piece_index=999)
    assert SearchExpert().action_values(observe(state)) == SearchExpert().action_values(
        observe(alternate)
    )


def test_unplaceable_preview_has_declared_penalty() -> None:
    state = new_game(0)
    board = ((1, 1, 1, 1, 0, 1, 1, 1, 1, 1),) + state.board[1:]
    observation = observe(replace(state, board=board, current="I", next_piece="O"))
    assert len(observation.legal_actions) == 1
    action = observation.legal_actions[0]
    _, cleared = place(board, "I", action)
    assert cleared == 0
    assert SearchExpert().action_values(observation) == {action.id: -10000}


def test_ties_follow_engine_order(monkeypatch) -> None:
    observation = observe(new_game(0))
    expert = SearchExpert()
    monkeypatch.setattr(
        expert, "action_values", lambda _: {a.id: 0 for a in observation.legal_actions}
    )
    assert expert.choose(observation).action_id == observation.legal_actions[0].id


def test_empty_actions_are_rejected() -> None:
    observation = replace(observe(new_game(0)), legal_actions=())
    assert SearchExpert().action_values(observation) == {}
    with pytest.raises(ValueError, match="no legal actions"):
        SearchExpert().choose(observation)
