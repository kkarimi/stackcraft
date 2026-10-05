"""CPU-only verification of three downloaded repositories against a trusted local manifest.

The trust anchor must come from independently retained local release preparation,
not from the same unverified download. This checks integrity against that anchor,
not publisher authenticity, model output parity, or whether a download was fresh.

Only these unlisted Hugging Face bookkeeping files are tolerated: root
.gitattributes; .cache/huggingface/{.gitignore,.gitignore.lock,CACHEDIR.TAG}; and
download metadata/empty locks corresponding to expected payloads or .gitattributes.
Incomplete downloads, upload metadata, arbitrary cache files and symlinks fail.
Ignored files are individually listed in the verification report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from stackcraft.data import audit_dataset

BUNDLES = ("model", "dataset", "demo")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
CACHE_TAG_PREFIX = "Signature: 8a477f597d28d172789f06886806bc55\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("manifest paths must be nonempty portable relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.as_posix() != value:
        raise ValueError("manifest paths must be canonical and cannot traverse directories")
    return path


def regular_input(path: Path, *, directory: bool = False) -> Path:
    # Resolve only after rejecting links in every supplied path component.
    absolute = path.absolute()
    if any(parent.is_symlink() for parent in (absolute, *absolute.parents)):
        raise ValueError(f"input paths cannot contain symlinks: {path}")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"expected {'directory' if directory else 'regular file'}: {path}")
    if not directory and not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"input must be a regular file: {path}")
    return path.resolve()


def _file_map(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("manifest files must be a nonempty mapping")
    for name, entry in value.items():
        safe_relative(name)
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes"}:
            raise ValueError("file manifest entries need exact sha256 and bytes fields")
        if not isinstance(entry["sha256"], str) or not HEX64.fullmatch(entry["sha256"]):
            raise ValueError("file SHA256 must be lowercase hexadecimal")
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise ValueError("file sizes must be nonnegative integers")
    return value


def _inventory(root: Path) -> tuple[dict[str, Path], set[str]]:
    files = {}
    directories = set()
    for folder, names, filenames in os.walk(root, followlinks=False):
        for name in (*names, *filenames):
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            safe_relative(relative)
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"download contains a symlink: {relative}")
            if stat.S_ISDIR(mode):
                directories.add(relative)
            elif stat.S_ISREG(mode):
                files[relative] = path
            else:
                raise ValueError(f"download contains a non-regular payload: {relative}")
    return files, directories


def _hf_metadata(relative: str, path: Path, expected: set[str]) -> bool:
    if relative == ".gitattributes":
        return True  # HF creates this repository file; it is not model/data code.
    if relative == ".cache/huggingface/.gitignore":
        return path.stat().st_size <= 2 and path.read_bytes() in (b"*", b"*\n")
    if relative == ".cache/huggingface/.gitignore.lock":
        return path.stat().st_size == 0
    if relative == ".cache/huggingface/CACHEDIR.TAG":
        if path.stat().st_size > 4096:
            return False
        text = path.read_text()
        return text.startswith(CACHE_TAG_PREFIX) and all(
            not line or line.startswith("#") for line in text.splitlines()[1:]
        )
    prefix = ".cache/huggingface/download/"
    if not relative.startswith(prefix):
        return False
    suffix = (
        ".metadata"
        if relative.endswith(".metadata")
        else ".lock"
        if relative.endswith(".lock")
        else None
    )
    if suffix is None:
        return False
    payload = relative[len(prefix) : -len(suffix)]
    if payload not in expected | {".gitattributes"}:
        return False
    if suffix == ".lock":
        return path.stat().st_size == 0
    if path.stat().st_size > 4096:
        return False
    lines = path.read_text().splitlines()
    if (
        len(lines) != 3
        or not HEX_COMMIT.fullmatch(lines[0])
        or not (HEX_COMMIT.fullmatch(lines[1]) or HEX64.fullmatch(lines[1]))
    ):
        return False
    try:
        timestamp = float(lines[2])
    except ValueError:
        return False
    return math.isfinite(timestamp) and timestamp >= 0


def _parent_directories(paths: set[str]) -> set[str]:
    return {
        parent.as_posix()
        for value in paths
        for parent in PurePosixPath(value).parents
        if parent.as_posix() != "."
    }


def verify_bundle(
    root: Path, expected: dict[str, dict[str, Any]], checkpoint: Any
) -> dict[str, Any]:
    actual, directories = _inventory(root)
    if missing := set(expected) - set(actual):
        raise ValueError(f"download is missing payloads: {sorted(missing)}")
    ignored = []
    for name in sorted(set(actual) - set(expected)):
        if not _hf_metadata(name, actual[name], set(expected)):
            raise ValueError(f"download contains unexpected payload: {name}")
        ignored.append(name)
    allowed_directories = _parent_directories(set(expected) | set(ignored))
    # HF can leave empty bookkeeping directories after finishing a download.
    allowed_directories.update({".cache", ".cache/huggingface", ".cache/huggingface/download"})
    allowed_directories.update(
        _parent_directories({f".cache/huggingface/download/{name}.metadata" for name in expected})
    )
    if extra_directories := directories - allowed_directories:
        raise ValueError(f"download contains unexpected directories: {sorted(extra_directories)}")
    for name, entry in expected.items():
        path = actual[name]
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"payload size mismatch: {name}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"payload SHA256 mismatch: {name}")
    if "release-files.json" not in expected:
        raise ValueError("trusted manifest must include each uploaded release-files.json")
    uploaded = json.loads(actual["release-files.json"].read_text())
    if (
        type(uploaded.get("schema_version")) is not int
        or uploaded.get("manifest_excludes_itself") is not True
    ):
        raise ValueError("uploaded release-files.json has invalid schema fields")
    _file_map(uploaded.get("files"))
    expected_uploaded = {
        "schema_version": 1,
        "manifest_excludes_itself": True,
        "checkpoint_sha256": checkpoint,
        "files": {key: value for key, value in expected.items() if key != "release-files.json"},
    }
    if uploaded != expected_uploaded:
        raise ValueError("uploaded release-files.json contradicts the trusted root manifest")
    return {
        "verified_files": len(expected),
        "verified_bytes": sum(entry["bytes"] for entry in expected.values()),
        "release_files_sha256": expected["release-files.json"]["sha256"],
        "ignored_huggingface_metadata": ignored,
    }


def verify_release(trusted_manifest: Path, roots: dict[str, Path], output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("verification output already exists; choose a new file")
    trusted_manifest = regular_input(trusted_manifest)
    if set(roots) != set(BUNDLES):
        raise ValueError("verification needs model, dataset and demo directories")
    roots = {name: regular_input(path, directory=True) for name, path in roots.items()}
    for name, root in roots.items():
        if any(
            root.is_relative_to(other) or other.is_relative_to(root)
            for key, other in roots.items()
            if key != name
        ):
            raise ValueError("download roots must be separate, non-overlapping directories")
        if output.resolve().is_relative_to(root):
            raise ValueError("verification output must be outside downloaded payloads")
    manifest = json.loads(trusted_manifest.read_text())
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("kind") != "local-release-bundle"
        or manifest.get("manifest_excludes_itself") is not True
    ):
        raise ValueError("unsupported trusted release manifest")
    files = _file_map(manifest.get("files"))
    expected: dict[str, dict[str, dict[str, Any]]] = {bundle: {} for bundle in BUNDLES}
    for relative, entry in files.items():
        path = safe_relative(relative)
        if len(path.parts) < 2 or path.parts[0] not in expected:
            raise ValueError("trusted release files must belong to model, dataset or demo")
        expected[path.parts[0]][PurePosixPath(*path.parts[1:]).as_posix()] = entry
    checkpoint = manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("trusted release needs a checkpoint hash mapping")
    for relative, digest in checkpoint.items():
        safe_relative(relative)
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise ValueError("trusted checkpoint SHA256 must be lowercase hexadecimal")
        if expected["model"].get(f"checkpoint/{relative}", {}).get("sha256") != digest:
            raise ValueError("trusted checkpoint hashes disagree with payload inventory")
    verified = {
        bundle: verify_bundle(
            roots[bundle], expected[bundle], checkpoint if bundle == "model" else None
        )
        for bundle in BUNDLES
    }
    dataset = roots["dataset"]
    if sha256(dataset / "manifest.json") != manifest.get("dataset_manifest_sha256"):
        raise ValueError("dataset manifest differs from the trusted dataset identity")
    dataset_manifest = json.loads((dataset / "manifest.json").read_text())
    records = {
        split: [json.loads(line) for line in (dataset / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "validation")
    }
    counts = audit_dataset(records, dataset_manifest)
    result = {
        "schema_version": 1,
        "status": "verified",
        "scope": "CPU payload integrity and dataset audit against caller-supplied trust anchor",
        "trusted_manifest_sha256": sha256(trusted_manifest),
        "bundles": verified,
        "dataset_counts": counts,
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "gpu_parity_checked": False,
        "publisher_authenticity_checked": False,
        "download_freshness_checked": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-manifest", type=Path, required=True)
    for bundle in BUNDLES:
        parser.add_argument(f"--{bundle}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = verify_release(
            args.trusted_manifest,
            {bundle: getattr(args, bundle) for bundle in BUNDLES},
            args.output,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {"status": report["status"], "output": str(args.output), "gpu_parity_checked": False}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
