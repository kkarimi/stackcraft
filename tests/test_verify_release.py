"""Verify separate downloaded roots using generated small dataset and toy payloads."""

import importlib.util
import json
from pathlib import Path

import pytest

from stackcraft.data import DatasetConfig, generate_dataset, write_dataset

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stackcraft_verify_release", ROOT / "scripts/verify_release.py"
)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def entry(path: Path) -> dict:
    return {"sha256": CLI.sha256(path), "bytes": path.stat().st_size}


@pytest.fixture
def release(tmp_path):
    roots = {name: tmp_path / f"downloaded-{name}" for name in CLI.BUNDLES}
    for root in roots.values():
        root.mkdir()
        (root / "README.md").write_text("Test-only downloaded payload.\n")
    model = roots["model"]
    (model / "checkpoint").mkdir()
    (model / "checkpoint/head.bin").write_bytes(b"test fixture, not model weights")
    checkpoint = {"head.bin": CLI.sha256(model / "checkpoint/head.bin")}
    bundle = generate_dataset(
        DatasetConfig(train_seeds=(10000,), validation_seeds=(20000,), max_pieces=2), "a" * 40
    )
    write_dataset(bundle, roots["dataset"])
    for name, root in roots.items():
        files = {
            path.relative_to(root).as_posix(): entry(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        (root / "release-files.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest_excludes_itself": True,
                    "files": files,
                    "checkpoint_sha256": checkpoint if name == "model" else None,
                }
            )
        )
    files = {
        f"{name}/{path.relative_to(root).as_posix()}": entry(path)
        for name, root in roots.items()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    trusted = tmp_path / "trusted-release-manifest.json"
    trusted.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "local-release-bundle",
                "manifest_excludes_itself": True,
                "files": files,
                "checkpoint_sha256": checkpoint,
                "dataset_manifest_sha256": CLI.sha256(roots["dataset"] / "manifest.json"),
            }
        )
    )
    return trusted, roots, {split: len(rows) for split, rows in bundle.records.items()}


def test_success_checks_separate_roots_and_uploaded_manifests(release, tmp_path):
    trusted, roots, counts = release
    output = tmp_path / "verification.json"
    report = CLI.verify_release(trusted, roots, output)
    assert report["status"] == "verified"
    assert report["trusted_manifest_sha256"] == CLI.sha256(trusted)
    assert report["dataset_counts"] == counts
    assert report["gpu_parity_checked"] is False
    assert report["publisher_authenticity_checked"] is False
    assert report["download_freshness_checked"] is False
    for bundle in CLI.BUNDLES:
        assert report["bundles"][bundle]["verified_files"] >= 2
        assert report["bundles"][bundle]["release_files_sha256"] == CLI.sha256(
            roots[bundle] / "release-files.json"
        )
    assert json.loads(output.read_text()) == report
    with pytest.raises(FileExistsError):
        CLI.verify_release(trusted, roots, output)


@pytest.mark.parametrize(
    "change",
    [
        "same_size_corruption",
        "changed_size",
        "missing",
        "additional",
        "manifest",
        "empty_directory",
    ],
)
def test_changed_missing_or_extra_payload_rejected(release, tmp_path, change):
    trusted, roots, _ = release
    path = roots["model"] / "README.md"
    if change == "same_size_corruption":
        path.write_bytes(b"X" * path.stat().st_size)
    elif change == "changed_size":
        path.write_text("different length")
    elif change == "missing":
        path.unlink()
    elif change == "additional":
        (roots["model"] / "unexpected.py").write_text("print('unexpected')")
    elif change == "empty_directory":
        (roots["model"] / "unlisted-directory").mkdir()
    else:
        (roots["model"] / "release-files.json").write_text("{}")
    output = tmp_path / "verification.json"
    with pytest.raises(ValueError):
        CLI.verify_release(trusted, roots, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "model/../escape",
        "model//README.md",
        "model/./README.md",
        "model\\evil",
        "model/C:evil",
    ],
)
def test_malicious_or_noncanonical_manifest_paths_rejected(release, tmp_path, path):
    trusted, roots, _ = release
    manifest = json.loads(trusted.read_text())
    manifest["files"][path] = {"sha256": "0" * 64, "bytes": 0}
    trusted.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        CLI.verify_release(trusted, roots, tmp_path / "verification.json")


@pytest.mark.parametrize("location", ["payload", "cache", "root", "parent"])
def test_symlinks_are_rejected_even_in_metadata(release, tmp_path, location):
    trusted, roots, _ = release
    if location == "payload":
        path = roots["demo"] / "README.md"
        path.unlink()
        path.symlink_to(roots["model"] / "README.md")
    elif location == "cache":
        (roots["demo"] / ".cache").symlink_to(tmp_path, target_is_directory=True)
    elif location == "root":
        link = tmp_path / "linked-root"
        link.symlink_to(roots["demo"], target_is_directory=True)
        roots["demo"] = link
    else:
        link = tmp_path / "linked-parent"
        link.symlink_to(tmp_path, target_is_directory=True)
        roots["demo"] = link / roots["demo"].name
    with pytest.raises(ValueError, match="symlink"):
        CLI.verify_release(trusted, roots, tmp_path / "verification.json")


def test_narrow_huggingface_metadata_exceptions_are_reported(release, tmp_path):
    trusted, roots, _ = release
    root = roots["model"]
    (root / ".gitattributes").write_text("*.bin filter=lfs diff=lfs merge=lfs -text\n")
    cache = root / ".cache/huggingface"
    (cache / "download/checkpoint").mkdir(parents=True)
    (cache / ".gitignore").write_text("*")
    (cache / "CACHEDIR.TAG").write_text(CLI.CACHE_TAG_PREFIX + "# generated by huggingface_hub\n")
    (cache / "download/checkpoint/head.bin.metadata").write_text(
        "a" * 40 + "\n" + "b" * 64 + "\n123.0\n"
    )
    (cache / "download/checkpoint/head.bin.lock").touch()
    report = CLI.verify_release(trusted, roots, tmp_path / "verification.json")
    ignored = report["bundles"]["model"]["ignored_huggingface_metadata"]
    assert len(ignored) == 5
    assert ".gitattributes" in ignored


def test_installed_huggingface_download_metadata_writer_is_accepted(release, tmp_path):
    local_folder = pytest.importorskip("huggingface_hub._local_folder")
    trusted, roots, _ = release
    path = roots["model"] / "checkpoint/head.bin"
    local_folder.write_download_metadata(
        roots["model"], "checkpoint/head.bin", "a" * 40, CLI.sha256(path)
    )
    report = CLI.verify_release(trusted, roots, tmp_path / "verification.json")
    ignored = report["bundles"]["model"]["ignored_huggingface_metadata"]
    assert ".cache/huggingface/download/checkpoint/head.bin.metadata" in ignored
    assert ".cache/huggingface/CACHEDIR.TAG" in ignored


@pytest.mark.parametrize(
    "name,content",
    [
        ("download/payload.py", "unexpected"),
        ("download/README.md.incomplete", "partial"),
        ("download/unlisted.metadata", "a" * 40 + "\n" + "b" * 64 + "\n123\n"),
        ("download/README.md.metadata", "not valid metadata"),
        ("download/README.md.lock", "not an empty lock"),
        ("upload/README.md.metadata", "upload bookkeeping is not a download"),
    ],
)
def test_arbitrary_cache_payloads_are_not_ignored(release, tmp_path, name, content):
    trusted, roots, _ = release
    path = roots["model"] / ".cache/huggingface" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    with pytest.raises(ValueError, match="unexpected payload"):
        CLI.verify_release(trusted, roots, tmp_path / "verification.json")


def test_dataset_schema_audited_even_when_corruption_is_rehashed_consistently(release, tmp_path):
    trusted, roots, _ = release
    dataset = roots["dataset"]
    rows = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
    rows[0]["action_id"] = "invalid-placement"
    from stackcraft.data import canonical_json

    (dataset / "train.jsonl").write_text("".join(canonical_json(row) + "\n" for row in rows))
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["splits"]["train"]["sha256"] = CLI.sha256(dataset / "train.jsonl")
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    uploaded = json.loads((dataset / "release-files.json").read_text())
    for name in ("train.jsonl", "manifest.json"):
        uploaded["files"][name] = entry(dataset / name)
    (dataset / "release-files.json").write_text(json.dumps(uploaded))
    trusted_json = json.loads(trusted.read_text())
    for name in ("train.jsonl", "manifest.json", "release-files.json"):
        trusted_json["files"][f"dataset/{name}"] = entry(dataset / name)
    trusted_json["dataset_manifest_sha256"] = CLI.sha256(dataset / "manifest.json")
    trusted.write_text(json.dumps(trusted_json))
    with pytest.raises(ValueError, match="teacher label"):
        CLI.verify_release(trusted, roots, tmp_path / "verification.json")


def test_output_cannot_modify_a_verified_download(release):
    trusted, roots, _ = release
    with pytest.raises(ValueError, match="outside downloaded"):
        CLI.verify_release(trusted, roots, roots["model"] / "verification.json")
