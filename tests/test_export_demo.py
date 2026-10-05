"""Use off-study toy episodes; export tests do not inspect held-out games."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from stackcraft.players import HeuristicPlayer, RandomPlayer
from stackcraft.tournament import run_tournament

TOY_SEED = 991337


@pytest.fixture
def exporter(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/export_demo.py"
    spec = importlib.util.spec_from_file_location("stackcraft_export_demo_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Production requires 30000; this seam keeps synthetic unit tests off test seeds.
    monkeypatch.setattr(module, "DEMO_SEED", TOY_SEED)
    return module


@pytest.fixture
def toy(tmp_path):
    result = run_tournament({"random": RandomPlayer, "heuristic": HeuristicPlayer}, [TOY_SEED], 3)
    path = tmp_path / "toy-tournament.json"
    path.write_text(json.dumps(result))
    return result, path


def test_export_preserves_real_replays_and_probabilities_with_source_hash(exporter, toy):
    original, path = toy
    result = exporter.export_manifest([path], ["heuristic", "random"], TOY_SEED)
    assert result["selection"] == {"seed": TOY_SEED, "method": "fixed-before-results"}
    assert result["sources"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [player["id"] for player in result["players"]] == ["heuristic", "random"]
    for player in result["players"]:
        episode = next(e for e in original["episodes"] if e["player_id"] == player["id"])
        assert player["replay"] == episode["replay"]
        assert player["outcome"] == episode["outcome"]
        assert player["decisions"][0]["probabilities"] == episode["decisions"][0]["probabilities"]
        assert "observation" not in player["decisions"][0]
    assert "no live model inference" in result["disclaimer"]


def test_individual_episode_inputs_supported_and_duplicates_rejected(exporter, toy, tmp_path):
    result, path = toy
    paths = []
    for index, episode in enumerate(result["episodes"]):
        episode_path = tmp_path / f"episode-{index}.json"
        episode_path.write_text(json.dumps(episode))
        paths.append(episode_path)
    assert len(exporter.export_manifest(paths, ["random", "heuristic"], TOY_SEED)["players"]) == 2
    with pytest.raises(ValueError, match="duplicate"):
        exporter.export_manifest([path, path], ["random", "heuristic"], TOY_SEED)


@pytest.mark.parametrize("corruption", ["outcome", "probability", "action", "observation", "cap"])
def test_export_rejects_inconsistent_evidence(exporter, toy, corruption):
    result, path = toy
    episode = result["episodes"][0]
    if corruption == "outcome":
        episode["outcome"]["lines"] += 1
    elif corruption == "probability":
        key = next(iter(episode["decisions"][0]["probabilities"]))
        episode["decisions"][0]["probabilities"][key] = -0.5
    elif corruption == "action":
        episode["decisions"][0]["action_id"] = "wrong"
    elif corruption == "observation":
        episode["decisions"][0]["observation"]["current"] = "wrong"
    elif corruption == "cap":
        episode["max_pieces"] = 4
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        exporter.export_manifest([path], ["random", "heuristic"], TOY_SEED)


def test_errors_are_preserved_instead_of_disguised_as_success(exporter, toy):
    result, path = toy
    original = result["episodes"][0]
    episode = copy.deepcopy(original)
    last = episode["decisions"].pop()
    episode["replay"]["actions"].pop()
    # Use the authoritative serializer after removing the final successful move.
    from stackcraft.replay import make_replay

    episode["replay"] = make_replay(TOY_SEED, episode["replay"]["actions"])
    failure = {"turn": 2, "kind": "player_error", "message": "test inference failed"}
    episode["errors"] = [failure]
    last.pop("action_id")
    last.pop("probabilities")
    last["error"] = failure
    episode["decisions"].append(last)
    episode["outcome"] = {**episode["replay"]["final"], "cap_hit": False, "status": "error"}
    result["episodes"][0] = episode
    path.write_text(json.dumps(result))
    exported = exporter.export_manifest([path], ["random", "heuristic"], TOY_SEED)
    assert exported["players"][0]["errors"] == [failure]
    assert exported["players"][0]["decisions"][-1]["error"] == failure


def test_fixed_seed_missing_player_and_existing_output_are_rejected(exporter, toy, tmp_path):
    _, path = toy
    with pytest.raises(ValueError, match="fixed"):
        exporter.export_manifest([path], ["random", "heuristic"], TOY_SEED + 1)
    with pytest.raises(ValueError, match="missing"):
        exporter.export_manifest([path], ["random", "trained"], TOY_SEED)
    destination = tmp_path / "existing.json"
    destination.write_text("do not replace")
    with pytest.raises(SystemExit):
        exporter.main(["--input", str(path), "--output", str(destination)])
    assert destination.read_text() == "do not replace"


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "http://example.com/report",
        "/report",
        "https:///",
        "https://user:pass@example.com/report",
        "https://example.com/ bad",
    ],
)
def test_report_url_rejects_unsafe_or_non_https_destinations(exporter, toy, url):
    _, path = toy
    with pytest.raises(ValueError):
        exporter.export_manifest([path], ["random", "heuristic"], TOY_SEED, report_url=url)


def test_optional_report_url_preserves_recorded_provenance(exporter, toy):
    _, path = toy
    original = exporter.export_manifest([path], ["random", "heuristic"], TOY_SEED)
    assert "report_url" not in original
    url = "https://example.com/study/results"
    linked = exporter.export_manifest([path], ["random", "heuristic"], TOY_SEED, report_url=url)
    assert linked.pop("report_url") == url
    assert linked == original
