"""Evaluate fixed Clef checkpoints without selecting from test results.

Examples (real GPU loads; run only after memory admission):
  uv run --locked --extra ml python scripts/evaluate_clef.py positions \
    --dataset data/study-v1 --checkpoint checkpoints/selected --output runs/validation
  uv run --locked --extra ml python scripts/evaluate_clef.py tournament \
    --checkpoint checkpoints/selected --seeds 0,1 --max-pieces 30 --output runs/dev-game

Final evaluation additionally requires --final-test --selection-file frozen.json.
The final selection schema is documented by validate_selection below.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stackcraft.clef import ENCODING_VERSION, MODEL_ID, MODEL_REVISION, ClefPlayer
from stackcraft.data import audit_dataset, canonical_json
from stackcraft.evaluation import FINAL_TEST_SEEDS, paired_report, summarize_positions
from stackcraft.players import (
    Decision,
    HeuristicPlayer,
    Player,
    RandomPlayer,
    observe,
    validate_decision,
)
from stackcraft.provenance import source_identity
from stackcraft.replay import replay_states
from stackcraft.schema import RULES_VERSION, GameState
from stackcraft.tournament import run_episode


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_hashes(path: Path) -> dict[str, str]:
    if (
        not (path / "training_config.json").is_file()
        or not (path / "joint_head.safetensors").is_file()
    ):
        raise ValueError("checkpoint must contain training_config.json and joint_head.safetensors")
    return {
        str(file.relative_to(path)): file_hash(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def parse_seeds(value: str) -> tuple[int, ...]:
    """Comma-separated integers or a Python-style start:stop range."""
    try:
        if ":" in value:
            start, stop = (int(part) for part in value.split(":"))
            seeds = tuple(range(start, stop))
        else:
            seeds = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers or start:stop"
        ) from error
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    return seeds


def validation_ineligibility(report: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    """The preregistered rule: all 215 rows, no errors, complete finite target NLL."""
    reasons = []
    metrics = report.get("players", {}).get("trained", {}).get("metrics", {})
    if report.get("mode") != "positions" or report.get("split") != "validation":
        reasons.append("not validation position evidence")
    if report.get("positions") != 215 or metrics.get("positions") != 215:
        reasons.append("not all 215 validation positions")
    if metrics.get("error_rate") != 0 or metrics.get("failed_predictions") != []:
        reasons.append("validation inference errors")
    if (
        metrics.get("complete_probability_coverage") is not True
        or metrics.get("probability_positions") != 215
        or metrics.get("missing_probabilities") != []
    ):
        reasons.append("incomplete validation probability coverage")
    nll = metrics.get("mean_nll")
    if (
        not isinstance(nll, (float, int))
        or isinstance(nll, bool)
        or not math.isfinite(nll)
        or metrics.get("nll_is_infinite") is not False
    ):
        reasons.append("nonfinite validation mean target NLL")
    expected_manifest = metadata.get("extra", {}).get("dataset_manifest_sha256")
    if not expected_manifest or report.get("dataset_manifest_sha256") != expected_manifest:
        reasons.append("checkpoint/evidence dataset manifest mismatch")
    return reasons


def validate_selection(args: argparse.Namespace, hashes: dict[str, str]) -> dict[str, Any] | None:
    """Require a concrete frozen protocol before any reserved test seed can run.

    JSON fields: schema_version=1, selection_split='validation', checkpoint_sha256
    (full relative file/hash mapping), validation_evidence={path,sha256},
    test_seeds (all 200), max_pieces, encoding_version, max_length,
    players, head_dtypes, bootstrap_samples=10000, bootstrap_seed=2026,
    max_error_rate=0.0, selected_key, decision_rule. Evidence paths are relative
    to the selection file. Native base head remains BF16; base-fp32 is an ablation.
    """
    overlap = set(args.seeds) & set(FINAL_TEST_SEEDS)
    if not args.final_test:
        if overlap:
            raise ValueError(
                "reserved test seeds require explicit --final-test and frozen selection"
            )
        if args.selection_file is not None:
            raise ValueError("--selection-file is reserved for --final-test")
        return None
    if tuple(args.seeds) != FINAL_TEST_SEEDS:
        raise ValueError("final evaluation must use exactly --seeds 30000:30200")
    if args.max_pieces != 200:
        raise ValueError("final evaluation is frozen at a 200-piece cap")
    if set(args.players) != {"base", "base-fp32", "trained", "random", "heuristic"}:
        raise ValueError("final evaluation requires all five preregistered players")
    if args.selection_file is None:
        raise ValueError("--final-test requires --selection-file")
    selection = json.loads(args.selection_file.read_text())
    required = {
        "schema_version": 1,
        "selection_split": "validation",
        "checkpoint_sha256": hashes,
        "test_seeds": list(FINAL_TEST_SEEDS),
        "test_seeds_sha256": hashlib.sha256(
            canonical_json(list(FINAL_TEST_SEEDS)).encode()
        ).hexdigest(),
        "max_pieces": args.max_pieces,
        "encoding_version": ENCODING_VERSION,
        "max_length": args.max_length,
        "players": args.players,
        "head_dtypes": {
            player: "bfloat16" if player == "base" else "float32"
            for player in args.players
            if player in ("base", "base-fp32", "trained")
        },
        "bootstrap_samples": 10000,
        "bootstrap_seed": 2026,
        "max_error_rate": 0.0,
    }
    for key, value in required.items():
        if type(selection.get(key)) is not type(value) or selection[key] != value:
            raise ValueError(f"frozen selection {key} differs from requested evaluation")
    if any(
        not isinstance(selection.get(key), str) or not selection[key]
        for key in ("selected_key", "decision_rule")
    ):
        raise ValueError("frozen selection needs selected_key and decision_rule")
    evidence = selection.get("validation_evidence", {})
    if not isinstance(evidence.get("path"), str) or not evidence["path"]:
        raise ValueError("frozen selection must identify validation evidence")
    path = args.selection_file.parent / evidence["path"]
    if not path.is_file() or file_hash(path) != evidence.get("sha256"):
        raise ValueError("validation evidence file/hash mismatch")
    validation = json.loads(path.read_text())
    if validation.get("mode") != "positions" or validation.get("split") != "validation":
        raise ValueError("selection evidence must be a validation-position evaluation")
    if validation.get("checkpoint_sha256") != hashes:
        raise ValueError("validation evidence evaluated a different checkpoint")
    if "trained" not in validation.get("players", {}):
        raise ValueError("validation evidence must include the trained checkpoint")
    if selection.get("validation_metrics") != validation["players"]["trained"].get("metrics"):
        raise ValueError("frozen validation metrics differ from evidence")
    metadata = json.loads((args.checkpoint / "training_config.json").read_text())
    if reasons := validation_ineligibility(validation, metadata):
        raise ValueError("selected checkpoint is ineligible: " + "; ".join(reasons))
    candidates = selection.get("candidates", [])
    if len(candidates) != 2 or [candidate.get("epoch") for candidate in candidates] != [1, 2]:
        raise ValueError("selection must preserve both preregistered epoch candidates")
    eligible = []
    for candidate in candidates:
        candidate_evidence = args.selection_file.parent / candidate["validation_evidence"]["path"]
        candidate_metadata = args.selection_file.parent / candidate["checkpoint_metadata"]["path"]
        if (
            file_hash(candidate_evidence) != candidate["validation_evidence"]["sha256"]
            or file_hash(candidate_metadata)
            != candidate["checkpoint_sha256"]["training_config.json"]
        ):
            raise ValueError("candidate evidence/metadata hash mismatch")
        report = json.loads(candidate_evidence.read_text())
        config = json.loads(candidate_metadata.read_text())
        if (
            report.get("checkpoint_sha256") != candidate["checkpoint_sha256"]
            or config.get("extra", {}).get("epoch") != candidate["epoch"]
        ):
            raise ValueError("candidate report/checkpoint identity mismatch")
        reasons = validation_ineligibility(report, config)
        if candidate.get("ineligibility_reasons") != reasons:
            raise ValueError("candidate eligibility differs from preserved evidence")
        if not reasons:
            eligible.append(
                (report["players"]["trained"]["metrics"]["mean_nll"], candidate["epoch"], candidate)
            )
    if not eligible:
        raise ValueError("no eligible validation checkpoint")
    winner = min(eligible, key=lambda item: (item[0], item[1]))[2]
    if winner["key"] != selection["selected_key"] or winner["checkpoint_sha256"] != hashes:
        raise ValueError("selection violates lowest validation NLL with earlier-epoch tie rule")
    return selection


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_player(name: str, args: argparse.Namespace, hashes: dict[str, str]) -> Player:
    if name == "random":
        return RandomPlayer()
    if name == "heuristic":
        return HeuristicPlayer()
    import torch

    from stackcraft.training import FP32DecisionHead, load_checkpoint

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False

    if args.device.startswith("cuda"):
        free, _ = torch.cuda.mem_get_info(args.device)
        if free < args.min_free_gib * 1024**3:
            raise RuntimeError(f"GPU has {free / 1024**3:.2f} GiB free; need {args.min_free_gib}")
    player = ClefPlayer.from_pretrained(
        trust_pinned_code=True,
        local_files_only=True,
        device=args.device,
        max_length=args.max_length,
    )
    if name == "trained":
        load_checkpoint(player.model, args.checkpoint, trainable=False)
        identity = hashlib.sha256(canonical_json(hashes).encode()).hexdigest()
        player.revision = (
            f"{MODEL_ID}@{MODEL_REVISION}:checkpoint-sha256-{identity}:{ENCODING_VERSION}"
        )
    elif name == "base-fp32":
        # Explicit precision ablation; the primary base keeps the native BF16 head.
        player.model.head = FP32DecisionHead(player.model.head)
        player.revision += ":head-fp32-ablation"
    player.model.eval()
    player.name = name
    player.runtime_config.update(
        {
            "revision": player.revision,
            "dtype": str(next(player.model.parameters()).dtype),
            "device": str(next(player.model.parameters()).device),
            "head_dtype": "bfloat16" if name == "base" else "float32",
            "evaluation_head": "native" if name == "base" else "fp32-gathered-rows",
            "versions": installed_versions(),
            "cuda_version": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
    )
    return player


def installed_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("torch", "transformers", "peft", "safetensors", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def validate_neural_runtimes(identities: dict[str, Any]) -> None:
    """Head precision may differ explicitly; shared backbone/input settings may not."""
    common = (
        "model_id",
        "base_revision",
        "encoding_version",
        "source_sha256",
        "max_length",
        "dtype",
        "device",
        "versions",
        "cuda_version",
        "cpu_threads",
        "float32_matmul_precision",
        "cuda_matmul_allow_tf32",
    )
    selected = [
        value["runtime_config"]
        for key, value in identities.items()
        if key in ("base", "base-fp32", "trained")
    ]
    if selected and any(
        {k: runtime.get(k) for k in common} != {k: selected[0].get(k) for k in common}
        for runtime in selected[1:]
    ):
        raise ValueError("neural players differ in shared backbone, encoding, precision or device")


def _release() -> None:
    gc.collect()
    import sys

    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    identity = source_identity(root)
    paths = [Path(__file__).resolve(), root / "uv.lock"] + [
        root / "src/stackcraft" / name
        for name in (
            "clef.py",
            "training.py",
            "evaluation.py",
            "tournament.py",
            "engine.py",
            "pieces.py",
            "players/__init__.py",
            "schema.py",
            "data.py",
            "provenance.py",
        )
    ]
    return {
        **identity,
        "source_commit": identity["source_commit"]
        + ("+working-tree" if identity["source_dirty"] else ""),
        "source_hashes": {str(path.relative_to(root)): file_hash(path) for path in paths},
        "rules_version": RULES_VERSION,
        "base_revision": MODEL_REVISION,
        "encoding_version": ENCODING_VERSION,
        "precision_policy": "native BF16 base; FP32 trained head; optional base-fp32 ablation",
        "installed_versions": installed_versions(),
    }


def evaluate_positions(args: argparse.Namespace, hashes: dict[str, str]) -> dict[str, Any]:
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    splits = {
        split: [
            json.loads(line) for line in (args.dataset / f"{split}.jsonl").read_text().splitlines()
        ]
        for split in ("train", "validation")
    }
    audit_dataset(splits, manifest)
    records = splits["validation"]
    if not records:
        raise ValueError("validation split is empty")
    result: dict[str, Any] = {
        "mode": "positions",
        "split": "validation",
        "positions": len(records),
        "dataset_manifest_sha256": file_hash(args.dataset / "manifest.json"),
        "checkpoint_sha256": hashes,
        "provenance": _provenance(),
        "players": {},
    }
    for name in args.players:
        loading = time.perf_counter()
        player = _load_player(name, args, hashes)
        load_seconds = time.perf_counter() - loading
        predictions: dict[str, Decision | None] = {}
        with (args.output / f"{name}-positions.jsonl").open("x") as stream:
            for row in records:
                raw = row["observation"]
                state = GameState(
                    tuple(tuple(r) for r in raw["board"]), 0, 0, raw["current"], raw["next_piece"]
                )
                observation = observe(state)
                event: dict[str, Any] = {"id": row["id"], "target_action_id": row["action_id"]}
                started = time.perf_counter()
                try:
                    decision = player.choose(observation)
                    validate_decision(decision, observation)
                except Exception as error:
                    predictions[row["id"]] = None
                    event["error"] = {"type": type(error).__name__, "message": str(error)}
                else:
                    predictions[row["id"]] = decision
                    event["decision"] = asdict(decision)
                    event["input_tokens"] = getattr(player, "last_input_tokens", None)
                event["decision_seconds"] = time.perf_counter() - started
                stream.write(canonical_json(event) + "\n")
                stream.flush()
        result["players"][name] = {
            "revision": player.revision,
            "runtime_config": getattr(player, "runtime_config", {}),
            "load_seconds": load_seconds,
            "metrics": summarize_positions(records, predictions),
            "predictions_file": f"{name}-positions.jsonl",
            "predictions_sha256": file_hash(args.output / f"{name}-positions.jsonl"),
        }
        validate_neural_runtimes(result["players"])
        _write(args.output / "report.json", result)
        del player
        _release()
    return result


def evaluate_tournament(
    args: argparse.Namespace, hashes: dict[str, str], selection: Any
) -> dict[str, Any]:
    episodes = []
    result: dict[str, Any] = {
        "mode": "tournament",
        "final_test": args.final_test,
        "seeds": list(args.seeds),
        "max_pieces": args.max_pieces,
        "checkpoint_sha256": hashes,
        "selection": selection,
        "provenance": _provenance(),
        "players": {},
        "episodes": episodes,
    }
    for name in args.players:
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        saved = {}
        for seed in args.seeds:
            path = directory / f"seed-{seed}.json"
            if path.exists():
                episode = json.loads(path.read_text())
                if (
                    episode.get("player_id") != name
                    or episode.get("seed") != seed
                    or episode.get("max_pieces") != args.max_pieces
                ):
                    raise ValueError("saved episode identity differs from frozen request")
                final = replay_states(episode["replay"])[-1]
                if (
                    any(
                        episode["outcome"][key] != getattr(final, key)
                        for key in ("score", "lines", "terminal")
                    )
                    or episode["outcome"]["pieces"] != final.piece_index
                ):
                    raise ValueError("saved episode outcome differs from replay")
                saved[seed] = episode
        metadata_path = directory / "player.json"
        if len(saved) == len(args.seeds):
            if not metadata_path.exists():
                raise ValueError("completed player episodes lack player metadata")
            metadata = json.loads(metadata_path.read_text())
            if any(
                episode["player"]["revision"] != metadata["revision"]
                or episode["player"].get("runtime_config", {}) != metadata["runtime_config"]
                for episode in saved.values()
            ):
                raise ValueError("completed episode identity differs from saved player metadata")
            result["players"][name] = metadata
            episodes.extend(saved[seed] for seed in args.seeds)
            continue
        loading = time.perf_counter()
        player = _load_player(name, args, hashes)
        load_seconds = time.perf_counter() - loading
        identity = {
            "revision": player.revision,
            "runtime_config": getattr(player, "runtime_config", {}),
        }
        if any(
            episode["player"]["revision"] != identity["revision"]
            or episode["player"].get("runtime_config", {}) != identity["runtime_config"]
            for episode in saved.values()
        ):
            raise ValueError("loaded player differs from saved episode identity")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if any(metadata.get(key) != value for key, value in identity.items()):
                raise ValueError("loaded player differs from saved player metadata")
            metadata["load_seconds"].append(load_seconds)
        else:
            metadata = {**identity, "load_seconds": [load_seconds]}
        result["players"][name] = metadata
        validate_neural_runtimes(result["players"])
        _write(metadata_path, metadata)
        # No excluded model calls. First-call effects remain in recorded latency.
        for seed in args.seeds:
            if seed in saved:
                episodes.append(saved[seed])
                continue
            if name == "random":
                player = RandomPlayer()  # episode-independent behavior RNG
            episode = run_episode(player, seed, args.max_pieces)
            episode["player_id"] = name
            _write(directory / f"seed-{seed}.json", episode)
            episodes.append(episode)
            _write(
                args.output / "progress.json",
                {"player": name, "seed": seed, "episodes": len(episodes)},
            )
        del player
        _release()
    validate_neural_runtimes(result["players"])
    if "trained" in args.players and "base" in args.players:
        result["trained_vs_base"] = paired_report(episodes, trained_id="trained", base_id="base")
    if "trained" in args.players and "heuristic" in args.players:
        result["trained_vs_heuristic"] = paired_report(
            episodes, trained_id="trained", base_id="heuristic"
        )
    if "trained" in args.players and "base-fp32" in args.players:
        result["trained_vs_base_fp32"] = paired_report(
            episodes, trained_id="trained", base_id="base-fp32"
        )
    if "base-fp32" in args.players and "base" in args.players:
        result["base_fp32_vs_base"] = paired_report(
            episodes, trained_id="base-fp32", base_id="base"
        )
    _write(args.output / "report.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="mode", required=True)
    for name in ("positions", "tournament"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--players",
            nargs="+",
            choices=("base", "base-fp32", "trained", "random", "heuristic"),
            default=["base", "base-fp32", "trained"]
            if name == "positions"
            else ["base", "base-fp32", "trained", "random", "heuristic"],
        )
        command.add_argument("--device", default="cuda")
        command.add_argument("--min-free-gib", type=float, default=22.0)
        command.add_argument("--max-length", type=int, default=4096)
        if name == "positions":
            command.add_argument("--dataset", type=Path, required=True)
        else:
            command.add_argument(
                "--seeds",
                type=parse_seeds,
                required=True,
                help="comma integers or start:stop (stop exclusive)",
            )
            command.add_argument("--max-pieces", type=int, required=True)
            command.add_argument("--final-test", action="store_true")
            command.add_argument("--selection-file", type=Path)
            command.add_argument(
                "--resume", action="store_true", help="resume matching atomic episode artifacts"
            )
    return root


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if len(set(args.players)) != len(args.players):
        cli.error("players must be unique")
    if args.max_length < 1 or not 0 < args.min_free_gib <= 1000:
        cli.error("max-length and min-free-gib must be positive finite values")
    if "trained" in args.players and args.checkpoint is None:
        cli.error("trained evaluation requires --checkpoint")
    if args.mode == "tournament" and args.max_pieces < 1:
        cli.error("max-pieces must be positive")
    try:
        hashes = checkpoint_hashes(args.checkpoint) if args.checkpoint is not None else {}
        selection = validate_selection(args, hashes) if args.mode == "tournament" else None
        request = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "resume"
        }
        provenance = _provenance()
        request.update(
            checkpoint_sha256=hashes,
            source_hashes=provenance["source_hashes"],
            installed_versions=provenance.get("installed_versions"),
        )
        request["selection_sha256"] = (
            file_hash(args.selection_file)
            if args.mode == "tournament" and args.selection_file
            else None
        )
        request = json.loads(canonical_json(request))
        if getattr(args, "resume", False):
            if json.loads((args.output / "request.json").read_text()) != request:
                raise ValueError("resume request/source/checkpoint differs from original run")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            _write(args.output / "request.json", request)
        if args.mode == "positions":
            result = evaluate_positions(args, hashes)
        else:
            result = evaluate_tournament(args, hashes, selection)
    except (ValueError, OSError) as error:
        cli.error(str(error))
    print(json.dumps({"mode": result["mode"], "output": str(args.output), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
