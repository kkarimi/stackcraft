"""Selection uses small fake checkpoints and synthetic reports, never model loads."""

import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from stackcraft.engine import new_game
from stackcraft.evaluation import summarize_positions
from stackcraft.players import Decision, observe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "stackcraft_select_cli", ROOT / "scripts/select_checkpoint.py"
    )
    assert SPEC is not None and SPEC.loader is not None
    CLI = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(CLI)
finally:
    sys.path.pop(0)


@pytest.fixture
def rows(monkeypatch):
    observation = observe(new_game(70))
    records = [
        {
            "id": f"validation-{index}",
            "action_id": observation.legal_actions[0].id,
            "observation": asdict(observation),
        }
        for index in range(215)
    ]
    monkeypatch.setattr(CLI, "load_validation", lambda path: (records, "manifest"))
    return records


def candidate(root, rows, epoch, target_probability, *, failure=False):
    checkpoint = root / f"epoch-{epoch:02d}"
    checkpoint.mkdir()
    metadata = {
        "extra": {
            "epoch": epoch,
            "config": CLI.TRAINING_CONFIG,
            "dataset_manifest_sha256": "manifest",
            "test_trajectories_used": False,
            "validation_used_for_training": False,
            "source_hashes": {"training": "same-source"},
            "dataset_counts": {"validation": 215},
            "dataset_split_sha256": {"validation": "same-validation"},
        }
    }
    (checkpoint / "training_config.json").write_text(json.dumps(metadata))
    (checkpoint / "joint_head.safetensors").write_bytes(b"FAKE-NOT-A-MODEL")
    hashes = CLI.checkpoint_hashes(checkpoint)
    validation = root / f"validation-{epoch}"
    validation.mkdir()
    predictions = {}
    events = []
    for index, row in enumerate(rows):
        target = row["action_id"]
        options = [action["id"] for action in row["observation"]["legal_actions"]]
        probs = {
            option: target_probability
            if option == target
            else (1 - target_probability) / (len(options) - 1)
            for option in options
        }
        decision = Decision(target, probs)
        event = {"id": row["id"], "target_action_id": target}
        if failure and index == 0:
            event["error"] = {"type": "RuntimeError", "message": "test"}
            predictions[row["id"]] = None
        else:
            event["decision"] = asdict(decision)
            predictions[row["id"]] = decision
        events.append(event)
    predictions_path = validation / "trained-positions.jsonl"
    predictions_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    metrics = summarize_positions(rows, predictions)
    report = {
        "mode": "positions",
        "split": "validation",
        "positions": 215,
        "dataset_manifest_sha256": "manifest",
        "checkpoint_sha256": hashes,
        "provenance": {
            "source_hashes": {"evaluation": "same"},
            "encoding_version": CLI.ENCODING_VERSION,
        },
        "players": {
            "trained": {
                "metrics": metrics,
                "runtime_config": {"head_dtype": "float32", "max_length": 4096},
                "predictions_file": predictions_path.name,
                "predictions_sha256": CLI.file_hash(predictions_path),
            }
        },
    }
    report_path = validation / "report.json"
    report_path.write_text(json.dumps(report))
    return checkpoint, report_path


@pytest.mark.parametrize(
    "probabilities,selected", [((0.2, 0.4), "epoch-02"), ((0.4, 0.4), "epoch-01")]
)
def test_lowest_nll_and_earlier_exact_tie_preserve_both_reports(
    tmp_path, rows, probabilities, selected
):
    pairs = [
        candidate(tmp_path, rows, epoch, probability)
        for epoch, probability in enumerate(probabilities, 1)
    ]
    output = tmp_path / "selection"
    result = CLI.select_checkpoint(pairs, tmp_path / "data", output)
    assert result["selected_key"] == selected
    assert result["validation_metrics"]["mean_nll"] == pytest.approx(-math.log(max(probabilities)))
    assert len(result["candidates"]) == 2
    for epoch in (1, 2):
        assert (output / f"epoch-{epoch:02d}/trained-positions.jsonl").is_file()
    assert (output / "selection-audit.json").is_file()
    assert result["max_pieces"] == 200
    assert len(result["test_seeds"]) == 200


def test_inference_error_makes_candidate_ineligible_even_with_better_remaining_nll(tmp_path, rows):
    pairs = [candidate(tmp_path, rows, 1, 0.9, failure=True), candidate(tmp_path, rows, 2, 0.2)]
    result = CLI.select_checkpoint(pairs, tmp_path / "data", tmp_path / "selection")
    assert result["selected_key"] == "epoch-02"
    assert "validation inference errors" in result["candidates"][0]["ineligibility_reasons"]


def test_both_nonfinite_candidates_leave_evidence_without_test_selection(tmp_path, rows):
    pairs = [candidate(tmp_path, rows, epoch, 0.0) for epoch in (1, 2)]
    output = tmp_path / "selection"
    with pytest.raises(ValueError, match="both epoch candidates"):
        CLI.select_checkpoint(pairs, tmp_path / "data", output)
    assert (output / "selection-audit.json").is_file()
    assert not (output / "selection.json").exists()


@pytest.mark.parametrize("corruption", ["metrics", "dataset", "extra_candidate", "provenance"])
def test_corrupted_or_unregistered_evidence_is_rejected(tmp_path, rows, corruption):
    pairs = [candidate(tmp_path, rows, epoch, 0.4) for epoch in (1, 2)]
    report = json.loads(pairs[1][1].read_text())
    if corruption == "metrics":
        report["players"]["trained"]["metrics"]["mean_nll"] = 0.0
    elif corruption == "dataset":
        report["dataset_manifest_sha256"] = "other"
    elif corruption == "provenance":
        report["provenance"]["source_hashes"] = {"evaluation": "changed"}
    else:
        pairs.append(pairs[0])
    pairs[1][1].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        CLI.select_checkpoint(pairs, tmp_path / "data", tmp_path / "selection")
