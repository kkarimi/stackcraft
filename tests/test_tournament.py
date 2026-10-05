import copy
import json

import pytest

from stackcraft.players import Decision, HeuristicPlayer, Observation, RandomPlayer
from stackcraft.replay import replay_states
from stackcraft.tournament import run_episode, run_tournament


def game_content(artifact: dict) -> dict:
    result = copy.deepcopy(artifact)
    result.pop("timing")
    for decision in result["decisions"]:
        decision.pop("decision_seconds")
    return result


@pytest.mark.parametrize("factory", [RandomPlayer, HeuristicPlayer])
def test_episode_repeatability_and_replay_parity(factory) -> None:
    episode = run_episode(factory(), seed=42, max_pieces=25)
    assert game_content(episode) == game_content(run_episode(factory(), 42, 25))
    decoded = json.loads(json.dumps(episode))
    final = replay_states(decoded["replay"])[-1]
    for name in ("lines", "score", "terminal"):
        assert episode["outcome"][name] == getattr(final, name)
    assert episode["outcome"]["pieces"] == final.piece_index
    assert episode["timing"]["count"] == len(episode["decisions"])
    assert not episode["errors"]


def test_tournament_pairs_seeds_and_fresh_players() -> None:
    calls = []

    def random_factory():
        calls.append(True)
        return RandomPlayer()

    tournament = run_tournament({"random": random_factory, "heuristic": HeuristicPlayer}, [5, 8], 5)
    assert len(calls) == 2
    for label in ("random", "heuristic"):
        group = [episode for episode in tournament["episodes"] if episode["player_id"] == label]
        assert [episode["seed"] for episode in group] == [5, 8]
        assert all(episode["outcome"]["cap_hit"] for episode in group)
    assert tournament["summary"]["random"]["error_rate"] == 0
    assert tournament["summary"]["heuristic"]["cap_hit_rate"] == 1
    for seed in (5, 8):
        pair = [e for e in tournament["episodes"] if e["seed"] == seed]
        streams = [
            [(d["observation"]["current"], d["observation"]["next_piece"]) for d in e["decisions"]]
            for e in pair
        ]
        assert streams[0] == streams[1]


class InvalidPlayer:
    name = "invalid"
    revision = "test"

    def choose(self, observation: Observation) -> Decision:
        return Decision("invalid")


class ThrowingPlayer(InvalidPlayer):
    def choose(self, observation: Observation) -> Decision:
        raise RuntimeError("unavailable")


@pytest.mark.parametrize(
    "player,kind", [(InvalidPlayer(), "invalid_decision"), (ThrowingPlayer(), "player_error")]
)
def test_failures_are_explicit_without_random_fallback(player, kind: str) -> None:
    episode = run_episode(player, 0)
    assert episode["outcome"]["status"] == "error"
    assert episode["outcome"]["pieces"] == 0
    assert episode["outcome"]["cap_hit"] is False
    assert episode["errors"][0]["kind"] == kind
    assert episode["replay"]["actions"] == []
    assert len(episode["decisions"]) == 1
    assert episode["timing"]["count"] == 1


def test_terminal_is_not_mislabeled_cap() -> None:
    episode = run_episode(RandomPlayer(), 0, 200)
    assert episode["outcome"]["terminal"]
    assert episode["outcome"]["status"] == "top_out"
    assert not episode["outcome"]["cap_hit"]


@pytest.mark.parametrize("seeds", [[], [1, 1], [True], [1.5]])
def test_bad_seed_pools_rejected(seeds) -> None:
    with pytest.raises(ValueError):
        run_tournament({"random": RandomPlayer}, seeds)


@pytest.mark.parametrize("limit", [0, -1, True, 1.1])
def test_bad_limits_rejected(limit) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        run_episode(RandomPlayer(), 0, limit)
