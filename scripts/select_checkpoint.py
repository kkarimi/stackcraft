"""Select only the two preregistered epochs using complete validation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from evaluate_clef import (
    ENCODING_VERSION,
    FINAL_TEST_SEEDS,
    checkpoint_hashes,
    file_hash,
    validate_selection,
    validation_ineligibility,
)

from stackcraft.data import audit_dataset, canonical_json
from stackcraft.evaluation import summarize_positions
from stackcraft.players import Decision

RULE = (
    "lowest finite mean target NLL over all 215 validation positions; "
    "exact tie chooses earlier epoch"
)
PLAYERS = ["base", "base-fp32", "trained", "random", "heuristic"]
TRAINING_CONFIG = {
    "mode": "lora",
    "epochs": 2,
    "seed": 42,
    "rank": 4,
    "learning_rate": 1e-5,
    "batch_size": 1,
    "gradient_accumulation": 8,
    "max_length": 4096,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "clip_gradient_norm": 1.0,
    "label_smoothing": 0.05,
    "brier_weight": 0.1,
}


def load_validation(dataset: Path) -> tuple[list[dict[str, Any]], str]:
    manifest = json.loads((dataset / "manifest.json").read_text())
    rows = {
        split: [json.loads(line) for line in (dataset / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "validation")
    }
    audit_dataset(rows, manifest)
    if len(rows["validation"]) != 215:
        raise ValueError("selection requires all 215 frozen validation positions")
    return rows["validation"], file_hash(dataset / "manifest.json")


def inspect_candidate(
    checkpoint: Path,
    report_path: Path,
    epoch: int,
    rows: list[dict[str, Any]],
    manifest_sha256: str,
) -> dict[str, Any]:
    hashes = checkpoint_hashes(checkpoint)
    metadata = json.loads((checkpoint / "training_config.json").read_text())
    report = json.loads(report_path.read_text())
    extra = metadata.get("extra", {})
    if extra.get("epoch") != epoch or extra.get("dataset_manifest_sha256") != manifest_sha256:
        raise ValueError("candidate epoch/dataset differs from preregistered study")
    if any(extra.get("config", {}).get(key) != value for key, value in TRAINING_CONFIG.items()):
        raise ValueError("candidate training configuration differs from preregistration")
    if (
        extra.get("test_trajectories_used") is not False
        or extra.get("validation_used_for_training") is not False
    ):
        raise ValueError("candidate training metadata must exclude test and validation use")
    if (
        report.get("checkpoint_sha256") != hashes
        or report.get("dataset_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("validation report is not bound to this checkpoint and dataset")
    if (
        report.get("positions") != 215
        or report.get("mode") != "positions"
        or report.get("split") != "validation"
    ):
        raise ValueError("candidate report must cover the complete validation split")
    trained = report.get("players", {}).get("trained")
    if not isinstance(trained, dict):
        raise ValueError("candidate report lacks trained predictions")
    predictions_path = report_path.parent / trained["predictions_file"]
    if file_hash(predictions_path) != trained["predictions_sha256"]:
        raise ValueError("raw prediction hash differs from validation report")
    events = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    if [event.get("id") for event in events] != [row["id"] for row in rows]:
        raise ValueError("raw predictions omit, duplicate, reorder or add validation positions")
    predictions = {}
    for row, event in zip(rows, events, strict=True):
        if event.get("target_action_id") != row["action_id"]:
            raise ValueError("prediction target differs from frozen teacher label")
        predictions[row["id"]] = None if "error" in event else Decision(**event["decision"])
    recomputed = summarize_positions(rows, predictions)
    if trained.get("metrics") != recomputed:
        raise ValueError("reported validation metrics differ from raw predictions")
    reasons = validation_ineligibility(report, metadata)
    return {
        "key": f"epoch-{epoch:02d}",
        "epoch": epoch,
        "checkpoint_sha256": hashes,
        "metrics": recomputed,
        "ineligibility_reasons": reasons,
        "checkpoint": checkpoint,
        "report_path": report_path,
        "report": report,
        "metadata": metadata,
    }


def select_checkpoint(
    candidates: list[tuple[Path, Path]], dataset: Path, output: Path
) -> dict[str, Any]:
    if len(candidates) != 2:
        raise ValueError("exactly epoch 01 and epoch 02 candidates are required")
    if output.exists():
        raise FileExistsError("selection output already exists")
    rows, manifest_hash = load_validation(dataset)
    inspected = [
        inspect_candidate(checkpoint, report, index + 1, rows, manifest_hash)
        for index, (checkpoint, report) in enumerate(candidates)
    ]
    first, second = inspected
    for key in ("source_hashes", "config", "dataset_split_sha256", "dataset_counts"):
        if first["metadata"]["extra"].get(key) != second["metadata"]["extra"].get(key):
            raise ValueError(f"candidate training provenance differs: {key}")
    for key in ("source_hashes", "installed_versions", "encoding_version", "base_revision"):
        if first["report"].get("provenance", {}).get(key) != second["report"].get(
            "provenance", {}
        ).get(key):
            raise ValueError(f"candidate evaluation provenance differs: {key}")
    runtime = [
        candidate["report"]["players"]["trained"]["runtime_config"] for candidate in inspected
    ]
    for key in (
        "model_id",
        "base_revision",
        "encoding_version",
        "source_sha256",
        "max_length",
        "dtype",
        "device",
        "head_dtype",
        "versions",
    ):
        if runtime[0].get(key) != runtime[1].get(key):
            raise ValueError(f"candidate evaluation runtime differs: {key}")
    output.mkdir(parents=True)
    preserved: list[dict[str, Any]] = []
    for candidate in inspected:
        folder = output / candidate["key"]
        folder.mkdir()
        shutil.copyfile(candidate["report_path"], folder / "report.json")
        shutil.copyfile(
            candidate["checkpoint"] / "training_config.json", folder / "training_config.json"
        )
        for player in candidate["report"]["players"].values():
            relative = Path(player["predictions_file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("prediction evidence must use a contained relative path")
            source = candidate["report_path"].parent / relative
            if file_hash(source) != player["predictions_sha256"]:
                raise ValueError("diagnostic prediction file hash mismatch")
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        preserved.append(
            {
                "key": candidate["key"],
                "epoch": candidate["epoch"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "checkpoint_origin": str(candidate["checkpoint"].resolve()),
                "checkpoint_metadata": {
                    "path": f"{candidate['key']}/training_config.json",
                    "sha256": file_hash(folder / "training_config.json"),
                },
                "validation_evidence": {
                    "path": f"{candidate['key']}/report.json",
                    "sha256": file_hash(folder / "report.json"),
                },
                "metrics": candidate["metrics"],
                "ineligibility_reasons": candidate["ineligibility_reasons"],
            }
        )
    eligible = [candidate for candidate in preserved if not candidate["ineligibility_reasons"]]
    audit = {
        "decision_rule": RULE,
        "dataset_manifest_sha256": manifest_hash,
        "candidates": preserved,
        "status": "selected" if eligible else "no-eligible-candidate",
    }
    (output / "selection-audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    if not eligible:
        raise ValueError(
            "both epoch candidates are ineligible; evidence preserved, no test selection"
        )
    selected = min(
        eligible, key=lambda candidate: (candidate["metrics"]["mean_nll"], candidate["epoch"])
    )
    selection = {
        "schema_version": 1,
        "selection_split": "validation",
        "selected_key": selected["key"],
        "decision_rule": RULE,
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "validation_metrics": selected["metrics"],
        "validation_evidence": selected["validation_evidence"],
        "candidates": preserved,
        "dataset_manifest_sha256": manifest_hash,
        "test_seeds": list(FINAL_TEST_SEEDS),
        "test_seeds_sha256": hashlib.sha256(
            canonical_json(list(FINAL_TEST_SEEDS)).encode()
        ).hexdigest(),
        "max_pieces": 200,
        "encoding_version": ENCODING_VERSION,
        "max_length": 4096,
        "players": PLAYERS,
        "head_dtypes": {"base": "bfloat16", "base-fp32": "float32", "trained": "float32"},
        "bootstrap_samples": 10000,
        "bootstrap_seed": 2026,
        "max_error_rate": 0.0,
    }
    selection_path = output / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    args = argparse.Namespace(
        final_test=True,
        seeds=FINAL_TEST_SEEDS,
        max_pieces=200,
        players=PLAYERS,
        max_length=4096,
        selection_file=selection_path,
        checkpoint=Path(selected["checkpoint_origin"]),
    )
    validate_selection(args, selected["checkpoint_sha256"])
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate01", nargs=2, type=Path, metavar=("CHECKPOINT", "REPORT"), required=True
    )
    parser.add_argument(
        "--candidate02", nargs=2, type=Path, metavar=("CHECKPOINT", "REPORT"), required=True
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/study-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        selection = select_checkpoint(
            [tuple(args.candidate01), tuple(args.candidate02)], args.dataset, args.output
        )
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "selected_key": selection["selected_key"],
                "mean_nll": selection["validation_metrics"]["mean_nll"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
