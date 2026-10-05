"""Train the fixed Stackcraft study for one or two epochs; no test-set access.

Requires the M4 feasibility gate and a GPU admitted by the parent workflow.
This script never stops services, rents compute, or selects a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from stackcraft.clef import ClefPlayer, encode_observation
from stackcraft.data import audit_dataset
from stackcraft.players import observe
from stackcraft.schema import GameState

STUDY_HASHES = {
    "train": "edd682761db95a4f25bb30a284489c54d9336a36da0a9b19d2cda860b428baa8",
    "validation": "eff9cdc5932e935959ac4d26dce6470d335090f7428a91930001954266d133bc",
}
STUDY_COUNTS = {"train": 827, "validation": 215}
TRAINING_SEED = 42
LORA_RANK = 4


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_study(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read/audit only the fixed train and validation files, never test trajectories."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = {}
    for split in ("train", "validation"):
        raw = (directory / f"{split}.jsonl").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != STUDY_HASHES[split]:
            raise ValueError(f"{split} file does not match the frozen study-v1 SHA256")
        records[split] = [json.loads(line) for line in raw.decode().splitlines()]
        if len(records[split]) != STUDY_COUNTS[split]:
            raise ValueError(f"{split} size differs from the frozen study-v1 count")
    audit_dataset(records, manifest)
    metadata = {
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dataset_split_sha256": dict(STUDY_HASHES),
        "dataset_counts": dict(STUDY_COUNTS),
        "dataset_source_commit": manifest["source_commit"],
        "dataset_config_sha256": manifest["config_sha256"],
        "test_trajectories_used": False,
        "validation_used_for_training": False,
    }
    return records["train"], metadata


def accumulation_groups(
    count: int, accumulation: int, *, epoch: int, seed: int = TRAINING_SEED
) -> list[tuple[int, ...]]:
    """Shuffle each complete epoch reproducibly and retain the final partial group."""
    if count < 1 or accumulation < 1 or epoch < 1:
        raise ValueError("count, accumulation and epoch must be positive")
    indices = list(range(count))
    random.Random(seed + epoch - 1).shuffle(indices)
    return [tuple(indices[start : start + accumulation]) for start in range(0, count, accumulation)]


def row_observation(row: dict[str, Any]):
    raw = row["observation"]
    return observe(
        GameState(tuple(tuple(r) for r in raw["board"]), 0, 0, raw["current"], raw["next_piece"])
    )


def train_epoch(
    player: Any,
    rows: list[dict[str, Any]],
    optimizer: Any,
    *,
    epoch: int,
    accumulation: int,
    mode: str,
    output: Path,
) -> dict[str, Any]:
    """Batch-one native training, averaging gradients over each actual group size."""
    import torch

    from stackcraft.training import decision_loss

    model = player.model
    model.train()
    if mode == "head":
        model.language_model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    groups = accumulation_groups(len(rows), accumulation, epoch=epoch)
    order = [rows[index]["id"] for group in groups for index in group]
    order_sha = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
    write_json(output / f"epoch-{epoch:02d}-order.json", {"row_ids": order, "sha256": order_sha})
    total_loss = 0.0
    microstep = 0
    started = time.monotonic()
    with (output / f"epoch-{epoch:02d}-events.jsonl").open("x") as log:
        for update, group in enumerate(groups, 1):
            optimizer.zero_grad(set_to_none=True)
            group_started = time.monotonic()
            for row_index in group:
                row = rows[row_index]
                step_started = time.monotonic()
                encoded = encode_observation(
                    row_observation(row),
                    player.processor.tokenizer,
                    player.native,
                    player.max_length,
                )
                batch = player.native.collate_records(
                    [encoded], player.processor.tokenizer.pad_token_id, device
                )
                logits = model(batch)[0][0]
                loss = decision_loss(logits, encoded, row["action_id"])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite training loss for {row['id']}")
                # The last group has three rows in study-v1; divide by three, not eight.
                (loss / len(group)).backward()
                if cuda:
                    torch.cuda.synchronize(device)
                value = float(loss.detach())
                total_loss += value
                microstep += 1
                event = {
                    "event": "microstep",
                    "epoch": epoch,
                    "microstep": microstep,
                    "optimizer_step": update,
                    "row_id": row["id"],
                    "tokens": len(encoded.input_ids),
                    "loss": value,
                    "accumulation_group_size": len(group),
                    "seconds": time.monotonic() - step_started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else 0,
                }
                log.write(json.dumps(event, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(event, allow_nan=False), flush=True)
                del loss, logits, batch
            # The global norm is nonfinite if any gradient is NaN or Inf. This also
            # checks accumulated gradients before clipping and before optimizer.step.
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if norm <= 0:
                raise RuntimeError("all trainable gradients are zero")
            optimizer.step()
            if cuda:
                torch.cuda.synchronize(device)
            event = {
                "event": "optimizer_step",
                "epoch": epoch,
                "optimizer_step": update,
                "microsteps": len(group),
                "gradient_norm_before_clip": float(norm),
                "seconds": time.monotonic() - group_started,
            }
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "epoch": epoch,
        "examples": microstep,
        "optimizer_steps": len(groups),
        "mean_training_loss": total_loss / microstep,
        "shuffle_order_sha256": order_sha,
        "seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else 0,
    }


def source_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    files = [
        Path(__file__).resolve(),
        root / "src/stackcraft/training.py",
        root / "src/stackcraft/clef.py",
        root / "src/stackcraft/data.py",
    ]
    return {
        "source_commit": commit,
        "source_dirty": bool(dirty),
        "source_hashes": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/study-v1"))
    parser.add_argument("--mode", choices=("lora", "head"), default="lora")
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args(argv)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.accumulation < 1 or args.max_length < 1:
        parser.error("--accumulation and --max-length must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    rows, dataset_metadata = load_study(args.dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "mode": args.mode,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": TRAINING_SEED,
        "rank": LORA_RANK if args.mode == "lora" else None,
        "batch_size": 1,
        "gradient_accumulation": args.accumulation,
        "max_length": args.max_length,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "clip_gradient_norm": 1.0,
        "label_smoothing": 0.05,
        "brier_weight": 0.1,
        "selection": "external validation only; this script does not choose a checkpoint",
    }
    metadata = {**dataset_metadata, **source_metadata(), "config": config}
    write_json(args.output / "run_config.json", metadata)
    report: dict[str, Any] = {"status": "running", "epochs": [], **metadata}
    write_json(args.output / "report.json", report)
    started = time.monotonic()
    try:
        import torch

        from stackcraft.training import parameter_hashes, prepare_trainable, save_checkpoint

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real study training run")
        free, total = torch.cuda.mem_get_info()
        if free < 25 * 1024**3:
            raise RuntimeError(f"requires at least 25 GiB free before loading; available={free}")
        random.seed(TRAINING_SEED)
        torch.manual_seed(TRAINING_SEED)
        torch.cuda.manual_seed_all(TRAINING_SEED)
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        report.update(
            gpu=torch.cuda.get_device_name(),
            initial_free_vram=free,
            total_vram=total,
            package_versions={
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "peft", "safetensors")
            },
        )
        player = ClefPlayer.from_pretrained(trust_pinned_code=True, max_length=args.max_length)
        prepare_trainable(player.model, mode=args.mode, rank=LORA_RANK)
        trainable_before = parameter_hashes(player.model, trainable=True)
        frozen_before = parameter_hashes(player.model, trainable=False)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in player.model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=0.01,
        )
        report["trainable_parameters"] = sum(
            parameter.numel() for parameter in player.model.parameters() if parameter.requires_grad
        )
        write_json(args.output / "report.json", report)
        for epoch in range(1, args.epochs + 1):
            torch.cuda.reset_peak_memory_stats()
            outcome = train_epoch(
                player,
                rows,
                optimizer,
                epoch=epoch,
                accumulation=args.accumulation,
                mode=args.mode,
                output=args.output,
            )
            checkpoint = args.output / f"epoch-{epoch:02d}"
            save_checkpoint(player.model, checkpoint, extra_metadata={**metadata, **outcome})
            # Fixed training positions are used only for serialization parity.
            # Validation selection remains external; no held-out test row is read.
            player.model.eval()
            reference_rows = rows[:4]
            write_json(
                checkpoint / "reference.json",
                {
                    "row_ids": [row["id"] for row in reference_rows],
                    "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
                    "dataset_train_sha256": metadata["dataset_split_sha256"]["train"],
                    "probabilities": [
                        player.choose(row_observation(row)).probabilities for row in reference_rows
                    ],
                    "absolute_tolerance": 1e-4,
                    "max_length": args.max_length,
                },
            )
            outcome["checkpoint"] = str(checkpoint)
            report["epochs"].append(outcome)
            write_json(args.output / "report.json", report)
        after = parameter_hashes(player.model, trainable=True)
        changed = [name for name in trainable_before if trainable_before[name] != after[name]]
        if not any(name.startswith("head.") for name in changed):
            raise RuntimeError("decision-head parameters did not change")
        if args.mode == "lora" and not any("lora_" in name for name in changed):
            raise RuntimeError("LoRA parameters did not change")
        if parameter_hashes(player.model, trainable=False) != frozen_before:
            raise RuntimeError("frozen backbone parameters changed")
        report.update(
            status="trained-awaiting-external-validation",
            changed_trainable_parameter_names=changed,
            frozen_parameters_unchanged=True,
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
