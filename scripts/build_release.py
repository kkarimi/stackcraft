"""Validate completed evidence and stage local release bundles; never upload anything."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from stackcraft.data import audit_dataset
from stackcraft.evaluation import FINAL_TEST_SEEDS, paired_report
from stackcraft.players import Decision, observe, validate_decision
from stackcraft.provenance import source_identity
from stackcraft.schema import GameState

ROOT = Path(__file__).resolve().parents[1]
PLAYERS = ("base", "base-fp32", "trained", "random", "heuristic")
COMPARISONS = {
    "trained_vs_base": ("trained", "base"),
    "trained_vs_heuristic": ("trained", "heuristic"),
    "trained_vs_base_fp32": ("trained", "base-fp32"),
    "base_fp32_vs_base": ("base-fp32", "base"),
}
CHECKPOINT_FILES = {
    "training_config.json",
    "joint_head.safetensors",
    "reference.json",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "adapter/README.md",
}
REPRO_SCRIPTS = (
    "benchmark_clef.py",
    "probe_clef_training.py",
    "train_clef.py",
    "evaluate_clef.py",
    "select_checkpoint.py",
    "export_demo.py",
    "build_release.py",
    "verify_release.py",
    "prepare_checkpoint.py",
    "gpu_session.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_helper(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"stackcraft_release_{name}", ROOT / "scripts" / name
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load reviewed release helper {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"release inputs must be regular files, not links: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def safe_relative(path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("release evidence paths must stay within their evidence directory")
    return relative


def validate_reference(checkpoint: Path, dataset: Path, records: dict[str, Any]) -> None:
    reference = json_file(checkpoint / "reference.json")
    rows = records["train"][:4]
    manifest_hash = sha256(dataset / "manifest.json")
    if (
        reference.get("row_ids") != [row["id"] for row in rows]
        or reference.get("dataset_manifest_sha256") != manifest_hash
        or reference.get("dataset_train_sha256") != sha256(dataset / "train.jsonl")
        or reference.get("absolute_tolerance") != 1e-4
    ):
        raise ValueError("checkpoint parity reference is not bound to the fixed training dataset")
    metadata = json_file(checkpoint / "training_config.json")
    if metadata.get("extra", {}).get("dataset_manifest_sha256") != manifest_hash:
        raise ValueError("checkpoint was trained with a different dataset manifest")
    config = metadata.get("extra", {}).get("config", {})
    if reference.get("max_length") != config.get("max_length"):
        raise ValueError("checkpoint reference context limit differs from training configuration")
    probabilities = reference.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != len(rows):
        raise ValueError("checkpoint needs all four fixed-row probability references")
    for row, distribution in zip(rows, probabilities, strict=True):
        raw = row["observation"]
        state = GameState(
            tuple(tuple(r) for r in raw["board"]), 0, 0, raw["current"], raw["next_piece"]
        )
        observation = observe(state)
        if not isinstance(distribution, dict) or not distribution:
            raise ValueError("checkpoint reference must contain complete probabilities")
        validate_decision(
            Decision(max(distribution, key=distribution.__getitem__), distribution), observation
        )


def validate_evaluation(
    evaluation: Path, selection_path: Path, checkpoint: Path, helpers: dict[str, ModuleType]
) -> tuple[dict[str, Any], dict[str, str], list[Path]]:
    evaluator, exporter = helpers["evaluate"], helpers["export"]
    hashes = evaluator.checkpoint_hashes(checkpoint)
    if not set(hashes).issubset(CHECKPOINT_FILES):
        raise ValueError("checkpoint contains unexpected files outside the release allowlist")
    if not {"training_config.json", "joint_head.safetensors", "reference.json"}.issubset(hashes):
        raise ValueError("checkpoint lacks trained head, metadata or reload reference")
    metadata = json_file(checkpoint / "training_config.json")
    if metadata.get("mode") == "lora" and not {
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "adapter/README.md",
    }.issubset(hashes):
        raise ValueError("LoRA checkpoint is missing adapter weights, configuration or attribution")
    selection = json_file(selection_path)
    report = json_file(evaluation / "report.json")
    request = json_file(evaluation / "request.json")
    if (
        report.get("mode") != "tournament"
        or report.get("final_test") is not True
        or report.get("seeds") != list(FINAL_TEST_SEEDS)
        or report.get("max_pieces") != 200
        or set(report.get("players", {})) != set(PLAYERS)
        or report.get("checkpoint_sha256") != hashes
        or report.get("selection") != selection
    ):
        raise ValueError("evaluation must be the complete frozen five-player, 200-seed final test")
    if (
        request.get("final_test") is not True
        or request.get("mode") != "tournament"
        or request.get("seeds") != list(FINAL_TEST_SEEDS)
        or request.get("max_pieces") != 200
        or set(request.get("players", [])) != set(PLAYERS)
        or request.get("checkpoint_sha256") != hashes
        or request.get("selection_sha256") != sha256(selection_path)
    ):
        raise ValueError("evaluation request is not bound to the checkpoint and frozen selection")
    args = SimpleNamespace(
        seeds=tuple(FINAL_TEST_SEEDS),
        final_test=True,
        max_pieces=200,
        players=request["players"],
        selection_file=selection_path,
        checkpoint=checkpoint,
        max_length=request["max_length"],
    )
    evaluator.validate_selection(args, hashes)
    episodes = report.get("episodes", [])
    expected_pairs = {(player, seed) for player in PLAYERS for seed in FINAL_TEST_SEEDS}
    by_pair = {(episode.get("player_id"), episode.get("seed")): episode for episode in episodes}
    if len(episodes) != len(expected_pairs) or set(by_pair) != expected_pairs:
        raise ValueError("final report must retain every player/seed pair exactly once")
    paths = [Path("report.json"), Path("request.json")]
    for player in PLAYERS:
        metadata_path = Path(player) / "player.json"
        metadata = json_file(evaluation / metadata_path)
        if metadata != report["players"][player]:
            raise ValueError("evaluation player metadata differs from its final report")
        paths.append(metadata_path)
        for seed in FINAL_TEST_SEEDS:
            relative = Path(player) / f"seed-{seed}.json"
            episode = json_file(evaluation / relative)
            if episode != by_pair[(player, seed)]:
                raise ValueError("episode file differs from final report")
            if episode["player"]["revision"] != metadata["revision"] or episode["player"].get(
                "runtime_config", {}
            ) != metadata.get("runtime_config", {}):
                raise ValueError("episode player identity differs from recorded metadata")
            # Recompute every board, visible observation, action and probability check.
            exporter.validated_player(episode, 0)
            paths.append(relative)
    evaluator.validate_neural_runtimes(report["players"])
    for key, (trained, base) in COMPARISONS.items():
        computed = paired_report(episodes, trained_id=trained, base_id=base)
        if computed != report.get(key):
            raise ValueError(f"reported {key} does not match recomputed paired outcomes")
    return report, hashes, paths


def selection_evidence_files(selection_path: Path) -> list[Path]:
    """Preserve self-contained selection evidence with original relative links."""
    selection = json_file(selection_path)
    files = {Path(selection_path.name)}
    pointers = [selection["validation_evidence"]]
    for candidate in selection["candidates"]:
        pointers.extend((candidate["validation_evidence"], candidate["checkpoint_metadata"]))
    for pointer in pointers:
        relative = safe_relative(pointer["path"])
        source = selection_path.parent / relative
        if sha256(source) != pointer["sha256"]:
            raise ValueError("selection evidence changed after its hash was frozen")
        files.add(relative)
        value = json_file(source)
        if value.get("mode") == "positions":
            for player in value["players"].values():
                predictions = relative.parent / safe_relative(player["predictions_file"])
                if sha256(selection_path.parent / predictions) != player["predictions_sha256"]:
                    raise ValueError("validation predictions do not match their recorded hash")
                files.add(predictions)
    if (selection_path.parent / "selection-audit.json").is_file():
        files.add(Path("selection-audit.json"))
    return sorted(files)


def render_model_card(template: str, report: dict[str, Any], selection: dict[str, Any]) -> str:
    summary = report["trained_vs_base"]
    rows = [
        "| Player | Lines mean (median) | Score mean (median) | "
        "Placed mean (median) | Cap hits | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    timing = [
        "| Player | Decision mean ms | Median ms | p95 ms | Invalid decisions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in PLAYERS:
        player = summary["players"][name]
        primary = player["failure_adjusted"]
        outcomes = " | ".join(
            f"{primary[metric]['mean']:.3f} ({primary[metric]['median']:.1f})"
            for metric in ("lines", "score", "pieces")
        )
        rows.append(
            f"| {name} | {outcomes} | {player['cap_hit_rate']:.1%} | {player['error_rate']:.1%} |"
        )
        latency = player["latency_seconds"]
        values = " | ".join(
            f"{latency[key] * 1000:.3f}" if latency[key] is not None else "unavailable"
            for key in ("mean", "median", "p95")
        )
        timing.append(f"| {name} | {values} | {player['invalid_decisions']} |")
    comparisons = [
        "| Comparison (first minus second) | Mean lines difference | Paired 95% interval |",
        "| --- | ---: | ---: |",
    ]
    for key, (first, second) in COMPARISONS.items():
        delta = report[key]["paired_trained_minus_base"]["lines"]
        comparisons.append(
            f"| {first} − {second} | {delta['mean_difference']:.3f} | "
            f"[{delta['ci95_lower']:.3f}, {delta['ci95_upper']:.3f}] |"
        )
    results = (
        "Verified final outcomes on all 200 paired seeds, with a 200-piece cap. "
        "Lines, score and placed-piece summaries use the preregistered zero-on-error policy. "
        "Cap hits indicate censored survival.\n\n"
        + "\n".join(rows)
        + "\n\nEach paired 95% bootstrap interval resamples complete episode differences. "
        "Trained versus native base is the primary comparison; the heuristic and FP32-head "
        "comparisons are separate checks. Intervals are not adjusted for multiple comparisons.\n\n"
        + "\n".join(comparisons)
        + "\n\nDecision latency includes recorded first-call effects, tokenization and policy "
        "overhead, but excludes game rendering and replay playback.\n\n"
        + "\n".join(timing)
        + "\n\nThe full reports retain raw outcomes, all failed seeds and probability logs."
    )
    card = template.replace(
        "**Release preparation: full-study results and selected checkpoint are pending.**",
        f"Selected checkpoint: **{selection['selected_key']}**, "
        "chosen on validation before final tests.",
    ).replace("**Insert verified held-out results here before publication.**", results)
    if "pending" in card.lower() or "insert verified" in card.lower():
        raise ValueError("model card still contains pending release text")
    return card + (
        "\n\n## Bundle layout\n\n"
        "`checkpoint/` preserves the selected checkpoint bytes. `code/` contains the source, "
        "locked environment, tests, scripts and milestone tutorials. `evidence/` contains the "
        "frozen selection, both validation candidates and complete final evaluation. "
        "From `code/`, run `uv sync --locked --extra ml`; load `../checkpoint` using the "
        "native Stackcraft loader. No access to a private GitHub repository is required.\n"
    )


def code_files(root: Path, *, demo: bool) -> list[Path]:
    """Explicit file classes; no planning, environments, arbitrary data, runs or credentials."""
    files = [
        Path(name)
        for name in (
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "README.md",
            "LICENSE",
            "NOTICE",
            "Dockerfile",
            ".dockerignore",
        )
    ]
    suffixes = {".py", ".html", ".js", ".css"}
    for path in (root / "src/stackcraft").rglob("*"):
        if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
            files.append(path.relative_to(root))
    files.append(Path("src/stackcraft/web/baseline-demo.json"))
    if not demo:
        files.extend(Path("scripts") / name for name in REPRO_SCRIPTS)
        files.extend((Path("release/model-card.md"), Path("release/dataset-card.md")))
        for folder, suffix in (("tests", ".py"), ("docs", ".md")):
            files.extend(path.relative_to(root) for path in (root / folder).rglob(f"*{suffix}"))
        for suffix in (".json", ".md"):
            files.extend(path.relative_to(root) for path in (root / "reports").glob(f"*{suffix}"))
        for milestone, name in enumerate(
            ("foundation", "game", "baselines", "data", "gpu-feasibility", "evaluation", "release")
        ):
            expected = Path("docs/tutorials") / f"{milestone:02d}-{name}.md"
            if expected not in files:
                raise ValueError(f"missing milestone tutorial: {expected}")
    return sorted(set(files))


def bundled_source_readme(readme: str) -> str:
    """Replace only the known private planning link with the shipped release tutorial."""
    return readme.replace(
        "[the project plan](plan.md)",
        "[the release tutorial](docs/tutorials/06-release.md)",
    )


def space_readme(report_url: str | None = None) -> str:
    """Describe the CPU demo without links to files excluded from its bundle."""
    readme = """---
title: Stackcraft
sdk: docker
app_port: 7860
license: apache-2.0
---

# Stackcraft

Play a simplified falling-block game and watch a recorded comparison of an
unchanged Clef model, a trained model and a heuristic on the same piece sequence.

**Human play is live. Bot comparisons are recorded.** The comparison replays saved
placements and their decision probabilities; it does not call a model while you
watch. Playback speed changes the animation, not measured inference latency.
The fixed demonstration sequence is seed30000. One game illustrates behavior;
it does not establish which player performs better across the complete study.

Choose an orientation and column, then drop the current piece vertically onto a
10×20 board. One next piece is visible. Completed rows clear simultaneously.
There is no timed gravity, hold, wall kick, tuck or T-spin bonus. This is a small
placement game, not a competitive Tetris implementation. Probabilities express
model preferences over legal placements, not calibrated chances of winning.

The Docker Space runs on CPU and needs no model weights, GPU or model credentials.
Human sessions are kept in memory; download a replay before a restart to retain
your game. The service runs as UID1000 on port7860 by default.

To run the same image locally:

```bash
docker build -t stackcraft-demo .
docker run --rm -p 7860:7860 stackcraft-demo
```

Open http://localhost:7860. The [Dockerfile](Dockerfile) and
[game source](src/stackcraft/) are included here. The
[recorded comparison](src/stackcraft/web/baseline-demo.json) contains the displayed
actions, probabilities, outcomes and source report hashes.

Apache-2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE). Stackcraft is independent
of Cloudflare, Qwen and Tetris.
"""
    if report_url:
        readme += f"\n[Complete study report](<{report_url}>).\n"
    return readme


def build_release(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError("release output already exists; use a new directory")
    helpers = {"evaluate": load_helper("evaluate_clef.py"), "export": load_helper("export_demo.py")}
    manifest = json_file(args.dataset / "manifest.json")
    records = {
        split: [
            json.loads(line) for line in (args.dataset / f"{split}.jsonl").read_text().splitlines()
        ]
        for split in ("train", "validation")
    }
    audit_dataset(records, manifest)
    for split in ("train", "validation"):
        if sha256(args.dataset / f"{split}.jsonl") != manifest["splits"][split]["sha256"]:
            raise ValueError("dataset file bytes differ from the manifest SHA256")
    if {split: len(rows) for split, rows in records.items()} != {"train": 827, "validation": 215}:
        raise ValueError("release dataset differs from the fixed 827/215-position study")
    validate_reference(args.checkpoint, args.dataset, records)
    report, checkpoint_hashes, evaluation_paths = validate_evaluation(
        args.evaluation, args.selection, args.checkpoint, helpers
    )
    selection = json_file(args.selection)
    evidence_paths = selection_evidence_files(args.selection)
    expected_demo = helpers["export"].export_manifest(
        [args.evaluation / "report.json"],
        ["base", "trained", "heuristic"],
        report_url=args.report_url,
    )
    if json_file(ROOT / "src/stackcraft/web/baseline-demo.json") != expected_demo:
        raise ValueError(
            "web demo must first be exported from this exact final report at fixed seed 30000"
        )
    card = render_model_card((ROOT / "release/model-card.md").read_text(), report, selection)
    dataset_card = (ROOT / "release/dataset-card.md").read_text()
    if "pending" in dataset_card.lower():
        raise ValueError("dataset card still contains pending release text")
    code_paths = code_files(ROOT, demo=False)
    demo_paths = code_files(ROOT, demo=True)
    identity = source_identity(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    for relative in checkpoint_hashes:
        copy_file(args.checkpoint / relative, args.output / "model/checkpoint" / relative)
    for relative in code_paths:
        copy_file(ROOT / relative, args.output / "model/code" / relative)
    source_readme = args.output / "model/code/README.md"
    source_readme.write_text(bundled_source_readme(source_readme.read_text()))
    for relative in demo_paths:
        copy_file(ROOT / relative, args.output / "demo" / relative)
    for relative in evaluation_paths:
        copy_file(args.evaluation / relative, args.output / "model/evidence/evaluation" / relative)
    for relative in evidence_paths:
        copy_file(
            args.selection.parent / relative, args.output / "model/evidence/selection" / relative
        )
    for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
        copy_file(args.dataset / name, args.output / "dataset" / name)
    for bundle in ("model", "dataset"):
        for name in ("LICENSE", "NOTICE"):
            copy_file(ROOT / name, args.output / bundle / name)
    (args.output / "model/README.md").write_text(card)
    (args.output / "dataset/README.md").write_text(dataset_card)
    (args.output / "demo/README.md").write_text(space_readme(args.report_url))
    copied_hashes = {
        relative: sha256(args.output / "model/checkpoint" / relative)
        for relative in checkpoint_hashes
    }
    if copied_hashes != checkpoint_hashes:
        raise ValueError("checkpoint bytes changed during staging")
    code = args.output / "model/code"
    write_json(
        code / "source-manifest.json",
        {
            "schema_version": 1,
            "source_commit": identity["source_commit"],
            "source_dirty": identity["source_dirty"],
            "source_hashes": {
                str(path.relative_to(code)): sha256(path)
                for path in sorted(code.rglob("*"))
                if path.is_file()
            },
        },
    )
    # Each independently uploaded repository carries its own byte manifest.
    # The root manifest additionally covers these manifests and all three bundles.
    for bundle in ("model", "dataset", "demo"):
        directory = args.output / bundle
        write_json(
            directory / "release-files.json",
            {
                "schema_version": 1,
                "files": {
                    str(path.relative_to(directory)): {
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in sorted(directory.rglob("*"))
                    if path.is_file()
                },
                "manifest_excludes_itself": True,
                "checkpoint_sha256": checkpoint_hashes if bundle == "model" else None,
            },
        )
    files = {
        str(path.relative_to(args.output)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(args.output.rglob("*"))
        if path.is_file()
    }
    result = {
        "schema_version": 1,
        "kind": "local-release-bundle",
        "published": False,
        "checkpoint_sha256": checkpoint_hashes,
        "selection_sha256": sha256(args.selection),
        "evaluation_report_sha256": sha256(args.evaluation / "report.json"),
        "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
        **identity,
        "files": files,
        "manifest_excludes_itself": True,
    }
    write_json(args.output / "release-manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("data/study-v1"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--report-url", help="Optional approved public report URL in the demo manifest"
    )
    args = parser.parse_args(argv)
    try:
        manifest = build_release(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Prepared {len(manifest['files'])} files locally in {args.output}; nothing published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
