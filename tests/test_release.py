"""Release staging checks on synthetic development fixtures; no held-out games."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from stackcraft.evaluation import paired_report
from stackcraft.players import RandomPlayer
from stackcraft.tournament import run_episode


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[1] / "scripts/build_release.py"
    spec = importlib.util.spec_from_file_location("stackcraft_test_release", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", ["/tmp/private", "../private", "nested/../../private", ""])
def test_evidence_paths_cannot_escape_selection_directory(script, path):
    with pytest.raises(ValueError, match="evidence paths"):
        script.safe_relative(path)


def test_file_copy_rejects_symlinks_and_preserves_exact_bytes(script, tmp_path):
    source = tmp_path / "original"
    source.write_bytes(b"\x00checkpoint unchanged\xff")
    destination = tmp_path / "model/checkpoint/weights"
    script.copy_file(source, destination)
    assert script.sha256(source) == script.sha256(destination)
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="not links"):
        script.copy_file(link, tmp_path / "copied-link")


def test_demo_allowlist_excludes_planning_credentials_and_arbitrary_data(script, tmp_path):
    for relative in (
        "src/stackcraft/server.py",
        "src/stackcraft/web/app.js",
        "src/stackcraft/web/baseline-demo.json",
        "src/stackcraft/web/secret.json",
        "src/stackcraft/web/.env",
        "plan.md",
        "record.md",
        "runs/raw.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    paths = {str(path) for path in script.code_files(tmp_path, demo=True)}
    assert {
        "src/stackcraft/server.py",
        "src/stackcraft/web/app.js",
        "Dockerfile",
        ".dockerignore",
    } <= paths
    assert "src/stackcraft/web/baseline-demo.json" in paths
    assert not paths & {
        "plan.md",
        "record.md",
        "runs/raw.json",
        "src/stackcraft/web/secret.json",
        "src/stackcraft/web/.env",
    }


def evaluation_fixture(script, tmp_path, monkeypatch):
    # Explicitly reduce only this test's validator pool to development seeds.
    # No policy is run on any reserved study seed.
    seeds = (7, 8)
    monkeypatch.setattr(script, "FINAL_TEST_SEEDS", seeds)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name, content in (
        ("training_config.json", '{"mode":"head"}'),
        ("joint_head.safetensors", "test bytes, not model weights"),
        ("reference.json", "{}"),
    ):
        (checkpoint / name).write_text(content)
    evaluator = script.load_helper("evaluate_clef.py")
    exporter = script.load_helper("export_demo.py")
    calls = []
    evaluator.validate_selection = lambda args, hashes: calls.append((args, hashes))
    hashes = evaluator.checkpoint_hashes(checkpoint)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    selection = tmp_path / "selection.json"
    selection.write_text('{"test_fixture":true}')
    prototypes = {seed: run_episode(RandomPlayer(), seed, 200) for seed in seeds}
    report = {
        "mode": "tournament",
        "final_test": True,
        "seeds": list(seeds),
        "max_pieces": 200,
        "checkpoint_sha256": hashes,
        "selection": script.json_file(selection),
        "players": {},
        "episodes": [],
    }
    for name in script.PLAYERS:
        metadata = {"revision": f"test-{name}", "runtime_config": {}, "load_seconds": [0.0]}
        report["players"][name] = metadata
        script.write_json(evaluation / name / "player.json", metadata)
        for seed in seeds:
            episode = copy.deepcopy(prototypes[seed])
            episode["player_id"] = name
            episode["player"] = {
                "name": name,
                "revision": metadata["revision"],
                "runtime_config": {},
            }
            report["episodes"].append(episode)
            script.write_json(evaluation / name / f"seed-{seed}.json", episode)
    for key, (trained, base) in script.COMPARISONS.items():
        report[key] = paired_report(report["episodes"], trained_id=trained, base_id=base)
    script.write_json(evaluation / "report.json", report)
    request = {
        "mode": "tournament",
        "final_test": True,
        "seeds": list(seeds),
        "max_pieces": 200,
        "checkpoint_sha256": hashes,
        "selection_sha256": script.sha256(selection),
        "players": list(script.PLAYERS),
        "max_length": 4096,
    }
    script.write_json(evaluation / "request.json", request)
    return evaluation, selection, checkpoint, {"evaluate": evaluator, "export": exporter}, calls


def test_complete_release_evidence_recomputes_replays_and_pairings(script, tmp_path, monkeypatch):
    evaluation, selection, checkpoint, helpers, calls = evaluation_fixture(
        script, tmp_path, monkeypatch
    )
    report, hashes, paths = script.validate_evaluation(evaluation, selection, checkpoint, helpers)
    assert len(report["episodes"]) == 10
    assert len(paths) == 17  # report/request + 5 metadata + 10 episodes.
    assert calls[0][0].checkpoint == checkpoint
    assert hashes == helpers["evaluate"].checkpoint_hashes(checkpoint)


@pytest.mark.parametrize("tampering", ["missing-pair", "outcome", "metric", "checkpoint"])
def test_release_rejects_incomplete_or_inconsistent_evidence(
    script, tmp_path, monkeypatch, tampering
):
    evaluation, selection, checkpoint, helpers, _ = evaluation_fixture(
        script, tmp_path, monkeypatch
    )
    report_path = evaluation / "report.json"
    report = script.json_file(report_path)
    if tampering == "missing-pair":
        report["episodes"].pop()
    elif tampering == "outcome":
        report["episodes"][0]["outcome"]["score"] = 999
        episode = report["episodes"][0]
        script.write_json(
            evaluation / episode["player_id"] / f"seed-{episode['seed']}.json", episode
        )
    elif tampering == "metric":
        report["trained_vs_base"]["paired_trained_minus_base"]["lines"]["mean_difference"] = 999
    else:
        (checkpoint / "joint_head.safetensors").write_text("changed")
    script.write_json(report_path, report)
    with pytest.raises(ValueError):
        script.validate_evaluation(evaluation, selection, checkpoint, helpers)


def test_model_card_uses_verified_results_and_rejects_other_pending_text(
    script, tmp_path, monkeypatch
):
    evaluation, _, _, _, _ = evaluation_fixture(script, tmp_path, monkeypatch)
    template = (
        "**Release preparation: full-study results and selected checkpoint are pending.**\n"
        "**Insert verified held-out results here before publication.**"
    )
    card = script.render_model_card(
        template, script.json_file(evaluation / "report.json"), {"selected_key": "epoch-01"}
    )
    assert "epoch-01" in card
    assert "| base-fp32 |" in card
    assert "paired 95% bootstrap interval" in card
    assert "pending" not in card.lower()
    with pytest.raises(ValueError, match="pending"):
        script.render_model_card(
            template + "\nAnother pending result",
            script.json_file(evaluation / "report.json"),
            {"selected_key": "epoch-01"},
        )


def test_existing_release_is_never_overwritten(script, tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        script.build_release(SimpleNamespace(output=tmp_path))
