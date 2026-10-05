"""Source identity for exact Git checkouts and published, hash-verified source bundles."""

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

SOURCE_MANIFEST = "source-manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(root: Path) -> dict[str, Any]:
    """Never mistake an enclosing, unrelated Git repository for this source tree.

    Published manifests identify original source and verify matching local bytes;
    they are provenance records, not signed proof of the claimed commit's authorship.
    """
    root = root.resolve()
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        if Path(top).resolve() == root:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("source requires a full 40-character Git commit")
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
            return {
                "source_commit": commit,
                "source_dirty": dirty,
                "source_identity_method": "exact-git-root",
            }
    except (OSError, subprocess.CalledProcessError):
        pass
    path = root / SOURCE_MANIFEST
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            "no exact Stackcraft Git root or bundled source-manifest.json; "
            "run from the released source directory or an initialized source checkout"
        )
    manifest = json.loads(path.read_text())
    commit = manifest.get("source_commit")
    hashes = manifest.get("source_hashes")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or not isinstance(commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", commit)
        or type(manifest.get("source_dirty")) is not bool
        or not isinstance(hashes, dict)
        or not hashes
    ):
        raise ValueError("invalid published source-manifest contract")
    changed = []
    for name, expected in hashes.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError("invalid source manifest path or SHA256")
        source = root / relative
        if (
            not source.is_file()
            or source.is_symlink()
            or not source.resolve().is_relative_to(root)
            or file_sha256(source) != expected
        ):
            changed.append(name)
    # Newly added importable files can change behavior even if existing files match.
    for folder in ("src", "scripts", "tests"):
        for source in (root / folder).rglob("*.py"):
            relative = str(source.relative_to(root))
            if relative not in hashes:
                changed.append(relative)
    return {
        "source_commit": commit,
        "source_dirty": manifest["source_dirty"] or bool(changed),
        "source_identity_method": "published-source-manifest",
        "source_manifest_sha256": file_sha256(path),
        "modified_source_files": sorted(set(changed)),
    }
