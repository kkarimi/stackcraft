"""Bounded real-weight GPU training/reload gate; never manages other workloads.

Run from the repository with uv run --locked --extra ml python scripts/probe_clef_training.py.
The parent process must impose a wall-clock timeout and restore borrowed GPU services.
"""

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from stackcraft.clef import ClefPlayer, encode_observation
from stackcraft.data import audit_dataset
from stackcraft.players import observe
from stackcraft.schema import GameState
from stackcraft.training import (
    decision_loss,
    load_checkpoint,
    parameter_hashes,
    prepare_trainable,
    save_checkpoint,
)


def observation(row):
    raw = row["observation"]
    return observe(
        GameState(tuple(tuple(r) for r in raw["board"]), 0, 0, raw["current"], raw["next_piece"])
    )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def probabilities(player, rows):
    return [player.choose(observation(row)).probabilities for row in rows]


def train_steps(player, rows, *, steps, learning_rate):
    model = player.model
    model.train()
    if model._stackcraft_training["mode"] == "head":
        model.language_model.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    events = []
    for index in range(steps):
        row = rows[index % len(rows)]
        encoded = encode_observation(
            observation(row), player.processor.tokenizer, player.native, player.max_length
        )
        batch = player.native.collate_records(
            [encoded], player.processor.tokenizer.pad_token_id, torch.device("cuda")
        )
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        logits = model(batch)[0][0]
        loss = decision_loss(logits, encoded, row["action_id"])
        if not torch.isfinite(loss):
            raise RuntimeError("training loss is nonfinite")
        loss.backward()
        gradient_sums = {"head": 0.0, "lora": 0.0}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                raise RuntimeError(f"nonfinite gradient: {name}")
            group = "lora" if "lora_" in name else "head"
            gradient_sums[group] += float(param.grad.detach().abs().sum())
        if gradient_sums["head"] <= 0:
            raise RuntimeError("no nonzero decision-head gradients")
        if any("lora_" in name for name, p in model.named_parameters() if p.requires_grad):
            if gradient_sums["lora"] <= 0:
                raise RuntimeError("no nonzero LoRA gradients")
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        event = {
            "step": index + 1,
            "row_id": row["id"],
            "tokens": len(encoded.input_ids),
            "loss": float(loss.detach()),
            "gradient_abs_sums": gradient_sums,
            "seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        events.append(event)
        print(json.dumps(event), flush=True)
    del optimizer
    model.zero_grad(set_to_none=True)
    model.eval()
    torch.cuda.empty_cache()
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/study-v1"))
    parser.add_argument("--reload", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = {"status": "running", "reload": bool(args.reload)}
    write_json(args.output / "report.json", report)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; this gate requires real GPU training")
        free, total = torch.cuda.mem_get_info()
        if free < 25 * 1024**3:
            raise RuntimeError(f"requires at least25GiB free before loading; available={free}")
        torch.manual_seed(42)
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        manifest_path = args.dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        records = {
            split: [
                json.loads(line)
                for line in (args.dataset / f"{split}.jsonl").read_text().splitlines()
            ]
            for split in ("train", "validation")
        }
        audit_dataset(records, manifest)
        # Development probe uses only the first four training positions, never test seeds.
        rows = records["train"][:4]
        report.update(
            gpu=torch.cuda.get_device_name(),
            total_vram=total,
            initial_free_vram=free,
            torch=torch.__version__,
            dataset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            row_ids=[row["id"] for row in rows],
        )
        player = ClefPlayer.from_pretrained(trust_pinned_code=True)
        torch.cuda.reset_peak_memory_stats()
        if args.reload:
            reference = json.loads((args.reload / "reference.json").read_text())
            for key in ("row_ids", "dataset_manifest_sha256"):
                if reference.get(key) != report[key]:
                    raise RuntimeError(f"reload reference {key} differs")
            player.model = load_checkpoint(player.model, args.reload)
            actual = probabilities(player, rows)
            if len(reference["probabilities"]) != len(actual):
                raise RuntimeError("reference record count differs")
            delta = 0.0
            for expected, observed in zip(reference["probabilities"], actual, strict=True):
                if expected.keys() != observed.keys():
                    raise RuntimeError("reload probability option set differs")
                delta = max(delta, *(abs(expected[k] - observed[k]) for k in expected))
            if delta > 1e-4:
                raise RuntimeError(f"fresh-process probability drift{delta} exceeds1e-4")
            report.update(max_absolute_probability_difference=delta, tolerance=1e-4)
        else:
            native_probabilities = probabilities(player, rows)
            report["native_probabilities"] = native_probabilities
            report["runtime_config"] = player.runtime_config
            report["load_and_native_seconds"] = time.monotonic() - started
            write_json(args.output / "report.json", report)
            prepare_trainable(player.model, mode="head")
            wrapped = probabilities(player, rows)
            report["fp32_head_initial_max_probability_drift"] = max(
                abs(a[key] - b[key])
                for a, b in zip(native_probabilities, wrapped, strict=True)
                for key in a
            )
            before_head = parameter_hashes(player.model, trainable=True)
            before_frozen = parameter_hashes(player.model, trainable=False)
            report["head_steps"] = train_steps(player, rows, steps=3, learning_rate=1e-5)
            if before_head == parameter_hashes(player.model, trainable=True):
                raise RuntimeError("head parameters did not change")
            if before_frozen != parameter_hashes(player.model, trainable=False):
                raise RuntimeError("frozen parameters changed during head training")
            save_checkpoint(
                player.model,
                args.output / "head-checkpoint",
                extra_metadata={"probe_rows": report["row_ids"]},
            )
            write_json(
                args.output / "head-checkpoint" / "reference.json",
                {
                    "probabilities": probabilities(player, rows),
                    "row_ids": report["row_ids"],
                    "dataset_manifest_sha256": report["dataset_manifest_sha256"],
                },
            )
            report["head_frozen_parameters_unchanged"] = True
            write_json(args.output / "report.json", report)
            del player
            gc.collect()
            torch.cuda.empty_cache()
            player = ClefPlayer.from_pretrained(trust_pinned_code=True)
            prepare_trainable(player.model, mode="lora", rank=4)
            before_lora = parameter_hashes(player.model, trainable=True)
            before_frozen = parameter_hashes(player.model, trainable=False)
            report["lora_steps"] = train_steps(player, rows, steps=5, learning_rate=1e-5)
            after_lora = parameter_hashes(player.model, trainable=True)
            if not any(
                "lora_" in name and value != after_lora[name] for name, value in before_lora.items()
            ):
                raise RuntimeError("LoRA parameters did not change")
            if before_frozen != parameter_hashes(player.model, trainable=False):
                raise RuntimeError("frozen parameters changed during LoRA training")
            checkpoint = args.output / "checkpoint"
            save_checkpoint(
                player.model, checkpoint, extra_metadata={"probe_rows": report["row_ids"]}
            )
            write_json(
                checkpoint / "reference.json",
                {
                    "probabilities": probabilities(player, rows),
                    "row_ids": report["row_ids"],
                    "dataset_manifest_sha256": report["dataset_manifest_sha256"],
                },
            )
            report["checkpoint"] = str(checkpoint)
            report["frozen_parameters_unchanged"] = True
        report.update(status="passed", elapsed_seconds=time.monotonic() - started)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
