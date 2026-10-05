"""Copy an immutable trained checkpoint and replace only its generated adapter card."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stackcraft.clef import MODEL_ID, MODEL_REVISION

CARD_PATH = "adapter/README.md"
REQUIRED_FILES = {
    "training_config.json",
    "joint_head.safetensors",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "reference.json",
    CARD_PATH,
}
ADAPTER_CARD = f"""---
base_model: {MODEL_ID}
library_name: peft
tags:
- lora
- stackcraft
---

# Stackcraft LoRA adapter component

This Stackcraft adapter adapts [{MODEL_ID}](https://huggingface.co/{MODEL_ID}),
released by Cloudflare and pinned at revision `{MODEL_REVISION}`.
The original model card and license are provided in that upstream repository.

This directory is only one component of the Stackcraft checkpoint. Load the
parent checkpoint's `joint_head.safetensors` and this adapter together through
Stackcraft's `load_checkpoint` function. The native decision head contains learned
parameters; loading only the LoRA adapter does not reproduce the trained model.
Use the exact pinned base and the parent `training_config.json` contract.

This component card does not claim improved performance or publication status.
See the release's full model card and evaluation report for measured results,
limitations, training provenance and reproduction commands. A prepared candidate
has not necessarily been selected by validation.
"""


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"symlinks are not accepted: {component}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(directory: Path) -> dict[str, str]:
    """Reject symlinks and special files; hash every regular file by relative path."""
    result = {}
    for path in sorted(directory.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"checkpoint contains a symlink or special file: {path}")
        if stat.S_ISREG(mode):
            result[path.relative_to(directory).as_posix()] = file_hash(path)
    return result


def prepare(source: Path, output: Path) -> dict[str, Any]:
    _reject_symlink_components(source.absolute())
    _reject_symlink_components(output.absolute())
    source = source.resolve()
    output = output.resolve()
    provenance = output.parent / f"{output.name}-preparation.json"
    _reject_symlink_components(provenance)
    if not source.is_dir():
        raise ValueError("source must be an existing checkpoint directory")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output directories must be separate, not nested")
    if output.exists() or provenance.exists():
        raise FileExistsError("output checkpoint or preparation record already exists")
    before = inventory(source)
    if not REQUIRED_FILES.issubset(before):
        missing = ", ".join(sorted(REQUIRED_FILES - before.keys()))
        raise ValueError(f"checkpoint is incomplete; missing: {missing}")
    if any((source / name).stat().st_size == 0 for name in REQUIRED_FILES):
        raise ValueError("required checkpoint files must not be empty")
    metadata = json.loads((source / "training_config.json").read_text())
    if (
        not isinstance(metadata, dict)
        or metadata.get("base_model") != MODEL_ID
        or metadata.get("base_revision") != MODEL_REVISION
        or metadata.get("mode") != "lora"
    ):
        raise ValueError("checkpoint does not identify the pinned Stackcraft LoRA base")
    for name in ("adapter/adapter_config.json", "reference.json"):
        if not isinstance(json.loads((source / name).read_text()), dict):
            raise ValueError(f"{name} must be a JSON object")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        for name in before:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination, follow_symlinks=False)
        # Recheck copied entries before writing the card, including a link introduced
        # by a source changing during copy. Never follow a copied symlink to write.
        inventory(output)
        (output / CARD_PATH).unlink()
        with (output / CARD_PATH).open("x") as stream:
            stream.write(ADAPTER_CARD)
        after = inventory(output)
        if before.keys() != after.keys():
            raise RuntimeError("prepared checkpoint changed the file inventory")
        changed = [name for name in before if before[name] != after[name]]
        if any(name != CARD_PATH for name in changed):
            raise RuntimeError("preparation altered a file other than adapter/README.md")
        if inventory(source) != before:
            raise RuntimeError("source checkpoint changed during preparation")
        report = {
            "schema_version": 1,
            "operation": "replace-generated-adapter-documentation-only",
            "created_at": datetime.now(UTC).isoformat(),
            "source": str(source),
            "output": str(output),
            "source_sha256": before,
            "output_sha256": after,
            "changed_files": changed,
            "unchanged_non_documentation": True,
            "source_unchanged": True,
            "base_model": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "preparation_script_sha256": file_hash(Path(__file__)),
        }
        with provenance.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except BaseException:
        # This call created the output exclusively; leave the raw source untouched.
        shutil.rmtree(output)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare(args.source, args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"output": report["output"], "changed_files": report["changed_files"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
