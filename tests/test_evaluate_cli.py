"""CLI contracts exercise only offline players; no GPU or neural-model loads."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stackcraft_evaluate_cli", ROOT / "scripts/evaluate_clef.py"
)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def test_seed_parser_and_reserved_seed_gate() -> None:
    assert CLI.parse_seeds("70,80") == (70, 80)
    assert CLI.parse_seeds("30000:30200") == CLI.FINAL_TEST_SEEDS
    with pytest.raises(argparse.ArgumentTypeError):
        CLI.parse_seeds("1,1")
    args = argparse.Namespace(seeds=(30000,), final_test=False, selection_file=None)
    with pytest.raises(ValueError, match="reserved test"):
        CLI.validate_selection(args, {})


def selection_fixture(tmp_path):
    metrics = {
        "positions": 215,
        "teacher_agreement": 0.5,
        "mean_nll": 1.0,
        "error_rate": 0.0,
        "failed_predictions": [],
        "complete_probability_coverage": True,
        "probability_positions": 215,
        "missing_probabilities": [],
        "nll_is_infinite": False,
    }
    candidates = []
    for epoch in (1, 2):
        checkpoint = tmp_path / f"checkpoint-{epoch}"
        checkpoint.mkdir()
        metadata = {"extra": {"epoch": epoch, "dataset_manifest_sha256": "manifest"}}
        (checkpoint / "training_config.json").write_text(json.dumps(metadata))
        (checkpoint / "joint_head.safetensors").write_bytes(b"fake-test-bytes-not-a-model")
        hashes = CLI.checkpoint_hashes(checkpoint)
        evidence = tmp_path / ("validation.json" if epoch == 1 else "validation-2.json")
        evidence.write_text(
            json.dumps(
                {
                    "mode": "positions",
                    "split": "validation",
                    "positions": 215,
                    "dataset_manifest_sha256": "manifest",
                    "checkpoint_sha256": hashes,
                    "players": {"trained": {"metrics": {**metrics, "mean_nll": float(epoch)}}},
                }
            )
        )
        candidates.append(
            {
                "key": f"epoch-{epoch:02d}",
                "epoch": epoch,
                "checkpoint_sha256": hashes,
                "checkpoint_metadata": {
                    "path": str(checkpoint.relative_to(tmp_path) / "training_config.json")
                },
                "validation_evidence": {"path": evidence.name, "sha256": CLI.file_hash(evidence)},
                "ineligibility_reasons": [],
            }
        )
    args = argparse.Namespace(
        seeds=CLI.FINAL_TEST_SEEDS,
        final_test=True,
        max_pieces=200,
        players=["base", "base-fp32", "trained", "random", "heuristic"],
        max_length=4096,
        selection_file=tmp_path / "selection.json",
        checkpoint=tmp_path / "checkpoint-1",
    )
    hashes = candidates[0]["checkpoint_sha256"]
    selection = {
        "schema_version": 1,
        "selection_split": "validation",
        "checkpoint_sha256": hashes,
        "test_seeds": list(CLI.FINAL_TEST_SEEDS),
        "test_seeds_sha256": hashlib.sha256(
            CLI.canonical_json(list(CLI.FINAL_TEST_SEEDS)).encode()
        ).hexdigest(),
        "max_pieces": 200,
        "encoding_version": CLI.ENCODING_VERSION,
        "max_length": 4096,
        "players": args.players,
        "head_dtypes": {"base": "bfloat16", "base-fp32": "float32", "trained": "float32"},
        "bootstrap_samples": 10000,
        "bootstrap_seed": 2026,
        "max_error_rate": 0.0,
        "selected_key": "epoch-01",
        "validation_metrics": metrics,
        "decision_rule": "lowest validation NLL",
        "candidates": candidates,
        "validation_evidence": candidates[0]["validation_evidence"],
    }
    args.selection_file.write_text(json.dumps(selection))
    return args, hashes, selection


def test_final_requires_matching_frozen_checkpoint_evidence_and_cap(tmp_path) -> None:
    args, hashes, selection = selection_fixture(tmp_path)
    assert CLI.validate_selection(args, hashes) == selection
    args.max_pieces = 100
    with pytest.raises(ValueError, match="200-piece"):
        CLI.validate_selection(args, hashes)
    args.max_pieces = 200
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        CLI.validate_selection(args, {"different": "checkpoint"})
    (tmp_path / "validation.json").write_text("changed")
    with pytest.raises(ValueError, match="evidence file/hash"):
        CLI.validate_selection(args, hashes)


def test_final_requires_precision_ablation_and_eligible_selected_checkpoint(tmp_path) -> None:
    args, hashes, selection = selection_fixture(tmp_path)
    args.players.remove("base-fp32")
    with pytest.raises(ValueError, match="all five"):
        CLI.validate_selection(args, hashes)
    report = json.loads((tmp_path / "validation.json").read_text())
    report["players"]["trained"]["metrics"]["mean_nll"] = None
    metadata = json.loads((args.checkpoint / "training_config.json").read_text())
    assert "nonfinite validation mean target NLL" in CLI.validation_ineligibility(report, metadata)


def test_completed_resume_checks_player_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(CLI, "_provenance", lambda: {"source_hashes": {"fake": "test"}})
    output = tmp_path / "run"
    args = [
        "tournament",
        "--players",
        "random",
        "--seeds",
        "70",
        "--max-pieces",
        "2",
        "--output",
        str(output),
    ]
    assert CLI.main(args) == 0
    path = output / "random/player.json"
    metadata = json.loads(path.read_text())
    metadata["revision"] = "changed-identity"
    path.write_text(json.dumps(metadata))
    with pytest.raises(SystemExit):
        CLI.main(args + ["--resume"])


def test_runtime_comparison_allows_declared_head_precision_but_not_encoding() -> None:
    native = {"encoding_version": "one", "dtype": "torch.bfloat16", "head_dtype": "bfloat16"}
    trained = {**native, "head_dtype": "float32"}
    identities = {"base": {"runtime_config": native}, "trained": {"runtime_config": trained}}
    CLI.validate_neural_runtimes(identities)
    trained["encoding_version"] = "two"
    with pytest.raises(ValueError, match="differ"):
        CLI.validate_neural_runtimes(identities)


def test_offline_tournament_resumes_without_replaying_completed_player(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(CLI, "_provenance", lambda: {"source_hashes": {"fake": "test"}})
    original = CLI._load_player
    calls = []

    def interrupted(name, args, hashes):
        calls.append(name)
        if name == "heuristic":
            raise RuntimeError("test interruption before second player")
        return original(name, args, hashes)

    output = tmp_path / "run"
    args = [
        "tournament",
        "--players",
        "random",
        "heuristic",
        "--seeds",
        "70,80",
        "--max-pieces",
        "2",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(CLI, "_load_player", interrupted)
    with pytest.raises(RuntimeError, match="test interruption"):
        CLI.main(args)
    first = (output / "random/seed-70.json").read_bytes()
    assert calls == ["random", "heuristic"]

    def resumed(name, args, hashes):
        assert name == "heuristic", "completed random player must not load again"
        return original(name, args, hashes)

    monkeypatch.setattr(CLI, "_load_player", resumed)
    assert CLI.main(args + ["--resume"]) == 0
    assert first == (output / "random/seed-70.json").read_bytes()
    report = json.loads((output / "report.json").read_text())
    assert len(report["episodes"]) == 4
    assert not report["final_test"]
    changed = args.copy()
    changed[changed.index("--max-pieces") + 1] = "3"
    with pytest.raises(SystemExit):
        CLI.main(changed + ["--resume"])


def test_reserved_cli_rejection_happens_before_player_load(tmp_path, monkeypatch) -> None:
    def forbidden(*args):
        raise AssertionError("must not load a player")

    monkeypatch.setattr(CLI, "_load_player", forbidden)
    with pytest.raises(SystemExit):
        CLI.main(
            [
                "tournament",
                "--players",
                "random",
                "--seeds",
                "30000",
                "--max-pieces",
                "200",
                "--output",
                str(tmp_path / "run"),
            ]
        )
    assert not (tmp_path / "run").exists()


def test_positions_evaluates_every_validation_row_and_persists_predictions(
    tmp_path, monkeypatch
) -> None:
    from dataclasses import asdict

    from stackcraft.engine import new_game
    from stackcraft.players import HeuristicPlayer, observe

    dataset = tmp_path / "data"
    dataset.mkdir()
    rows = []
    for index in range(3):
        observation = observe(new_game(70 + index))
        rows.append(
            {
                "id": str(index),
                "observation": asdict(observation),
                "action_id": HeuristicPlayer().choose(observation).action_id,
            }
        )
    (dataset / "manifest.json").write_text("{}")
    (dataset / "train.jsonl").write_text("")
    (dataset / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(CLI, "audit_dataset", lambda *args: None)
    monkeypatch.setattr(CLI, "_provenance", lambda: {"source_hashes": {"fake": "test"}})
    monkeypatch.setattr(CLI, "_release", lambda: None)
    output = tmp_path / "positions"
    assert (
        CLI.main(
            [
                "positions",
                "--dataset",
                str(dataset),
                "--players",
                "heuristic",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert report["positions"] == 3
    assert report["players"]["heuristic"]["metrics"]["teacher_agreement"] == 1
    predictions = [
        json.loads(line) for line in (output / "heuristic-positions.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in predictions] == ["0", "1", "2"]
