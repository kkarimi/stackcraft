import copy
import json

import pytest

from stackcraft.engine import legal_actions, new_game, step
from stackcraft.replay import make_replay, replay_states


def episode(seed: int = 123) -> tuple[list[str], list]:
    states = [new_game(seed)]
    actions = []
    while not states[-1].terminal:
        action = legal_actions(states[-1])[0].id
        actions.append(action)
        states.append(step(states[-1], action).state)
    return actions, states


def test_json_roundtrip_recreates_every_state_and_terminal_outcome() -> None:
    actions, states = episode()
    artifact = make_replay(123, actions)
    assert replay_states(json.loads(json.dumps(artifact))) == states
    assert artifact["final"]["terminal"] is True
    assert artifact["final"]["pieces"] == len(actions)
    actions.clear()
    assert artifact["actions"]


def test_empty_and_partial_games_are_valid_replays() -> None:
    assert replay_states(make_replay(1, [])) == [new_game(1)]
    actions, states = episode()
    assert replay_states(make_replay(123, actions[:3])) == states[:4]


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("rules_version", "unknown"),
        ("seed", True),
        ("seed", "123"),
        ("actions", "r0x0"),
        ("actions", [4]),
        ("actions", ["invalid"]),
        ("final", {}),
        ("final", None),
    ],
)
def test_malformed_or_tampered_metadata_is_rejected(key: str, value: object) -> None:
    artifact = make_replay(123, [])
    artifact[key] = value
    with pytest.raises(ValueError):
        replay_states(artifact)


@pytest.mark.parametrize("field", ["score", "lines", "pieces", "terminal"])
def test_tampered_final_outcomes_are_rejected(field: str) -> None:
    actions, _ = episode()
    artifact = make_replay(123, actions)
    artifact["final"][field] = not artifact["final"][field] if field == "terminal" else 999
    with pytest.raises(ValueError, match="summary"):
        replay_states(artifact)


def test_boolean_cannot_impersonate_numeric_summary() -> None:
    artifact = make_replay(123, [])
    artifact["final"]["score"] = False
    with pytest.raises(ValueError, match="summary"):
        replay_states(artifact)


def test_actions_after_game_over_are_rejected() -> None:
    actions, _ = episode()
    artifact = make_replay(123, actions)
    artifact["actions"].append("r0x0")
    with pytest.raises(ValueError, match="invalid replay move"):
        replay_states(artifact)


def test_summary_optional_and_validation_does_not_mutate_artifact() -> None:
    actions, states = episode()
    artifact = make_replay(123, actions)
    del artifact["final"]
    before = copy.deepcopy(artifact)
    assert replay_states(artifact) == states
    assert artifact == before
