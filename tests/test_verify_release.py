"""Verify separate downloaded roots using generated small dataset and toy payloads."""

import hashlib
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
    assert report["skipped_bundles"] == []
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


@pytest.mark.parametrize("skip_demo", [False, True])
def test_dataset_schema_audited_even_when_corruption_is_rehashed_consistently(
    release, tmp_path, skip_demo
):
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
    if skip_demo:
        roots = {name: root for name, root in roots.items() if name != "demo"}
    with pytest.raises(ValueError, match="teacher label"):
        CLI.verify_release(trusted, roots, tmp_path / "verification.json", skip_demo=skip_demo)


def test_output_cannot_modify_a_verified_download(release):
    trusted, roots, _ = release
    with pytest.raises(ValueError, match="outside downloaded"):
        CLI.verify_release(trusted, roots, roots["model"] / "verification.json")


def write_tree_fixture(trusted, root, *, lfs=False):
    """Make the documented format with revision/etag metadata for every payload."""
    commit = "a" * 40
    expected = {
        name.removeprefix("model/"): value
        for name, value in json.loads(trusted.read_text())["files"].items()
        if name.startswith("model/")
    }
    files = {}
    for name, value in expected.items():
        raw = (root / name).read_bytes()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        info = {"size": len(raw), "blob_id": blob}
        etag = blob
        if lfs and name == "checkpoint/head.bin":
            info.update(lfs_sha256=value["sha256"], lfs_size=len(raw), xet_hash="B" * 64)
            etag = value["sha256"]
        files[name] = info
        metadata = root / f".cache/huggingface/download/{name}.metadata"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(f"{commit}\n{etag}\n123.0\n")
    path = root / f".cache/huggingface/trees/{commit}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format_version": 1, "files": files}))
    return path


@pytest.mark.parametrize("lfs", [False, True])
def test_full_revision_bound_tree_with_git_or_lfs_hashes_is_accepted(release, tmp_path, lfs):
    trusted, roots, _ = release
    tree = write_tree_fixture(trusted, roots["model"], lfs=lfs)
    report = CLI.verify_release(trusted, roots, tmp_path / "verified.json")
    ignored = report["bundles"]["model"]["ignored_huggingface_metadata"]
    assert tree.relative_to(roots["model"]).as_posix() in ignored


@pytest.mark.parametrize(
    "change",
    [
        "extra_payload",
        "missing_payload",
        "traversal",
        "unknown_field",
        "size",
        "blob_hash",
        "lfs_hash",
        "lfs_size",
        "xet_hash",
        "revision",
        "etag",
        "missing_metadata",
        "duplicate_json_key",
        "bool_version",
        "oversized",
        "wrong_filename",
    ],
)
def test_tree_cache_cannot_hide_unexpected_or_mismatched_content(release, tmp_path, change):
    trusted, roots, _ = release
    root = roots["model"]
    tree = write_tree_fixture(trusted, root, lfs=True)
    data = json.loads(tree.read_text())
    if change == "extra_payload":
        data["files"]["unlisted.py"] = data["files"]["README.md"]
    elif change == "missing_payload":
        del data["files"]["README.md"]
    elif change == "traversal":
        data["files"]["../README.md"] = data["files"].pop("README.md")
    elif change == "unknown_field":
        data["files"]["README.md"]["unexpected"] = "arbitrary hidden data"
    elif change == "size":
        data["files"]["README.md"]["size"] += 1
    elif change == "blob_hash":
        data["files"]["README.md"]["blob_id"] = "0" * 40
    elif change == "lfs_hash":
        data["files"]["checkpoint/head.bin"]["lfs_sha256"] = "0" * 64
    elif change == "lfs_size":
        data["files"]["checkpoint/head.bin"]["lfs_size"] += 1
    elif change == "xet_hash":
        data["files"]["checkpoint/head.bin"]["xet_hash"] = "not a hash"
    elif change == "revision":
        tree = tree.rename(tree.with_name("b" * 40 + ".json"))
    elif change == "etag":
        metadata = root / ".cache/huggingface/download/README.md.metadata"
        metadata.write_text("a" * 40 + "\n" + "0" * 40 + "\n123.0\n")
    elif change == "missing_metadata":
        (root / ".cache/huggingface/download/README.md.metadata").unlink()
    elif change == "bool_version":
        data["format_version"] = True
    elif change == "wrong_filename":
        tree = tree.rename(tree.with_name("arbitrary.json"))
    tree.write_text(json.dumps(data))
    if change == "duplicate_json_key":
        tree.write_text(
            tree.read_text().replace(
                '"format_version": 1', '"format_version": 1, "format_version": 1'
            )
        )
    elif change == "oversized":
        tree.write_text(tree.read_text() + " " * 20000)
    with pytest.raises(ValueError, match="unexpected payload"):
        CLI.verify_release(trusted, roots, tmp_path / "verified.json")


def test_installed_huggingface_tree_writer_is_accepted(release, tmp_path):
    tree_cache = pytest.importorskip("huggingface_hub._tree_cache")
    trusted, roots, _ = release
    tree = write_tree_fixture(trusted, roots["model"], lfs=True)
    data = json.loads(tree.read_text())
    entries = {
        name: tree_cache.TreeCacheEntry.from_json(info) for name, info in data["files"].items()
    }
    tree.unlink()
    tree_cache.write_tree_cache(str(tree.parent.parent), "a" * 40, entries)
    report = CLI.verify_release(trusted, roots, tmp_path / "verified.json")
    assert (
        tree.relative_to(roots["model"]).as_posix()
        in report["bundles"]["model"]["ignored_huggingface_metadata"]
    )


def test_actual_downloaded_dataset_tree_cache_integration():
    """Read-only local integration; public downloads are absent on clean checkouts."""
    trusted = ROOT / "runs/release-public-v1/release-manifest.json"
    root = ROOT / "runs/download-public-v1/dataset"
    if not trusted.is_file() or not (root / ".cache/huggingface/trees").is_dir():
        pytest.skip("actual revision-pinned release download is not present")
    expected = {
        name.removeprefix("dataset/"): value
        for name, value in json.loads(trusted.read_text())["files"].items()
        if name.startswith("dataset/")
    }
    report = CLI.verify_bundle(root, expected, None)
    assert report["verified_files"] == len(expected)
    assert any("/trees/" in name for name in report["ignored_huggingface_metadata"])


def test_explicit_skip_demo_verifies_only_model_and_dataset(release, tmp_path):
    trusted, roots, counts = release
    # A bad demo is deliberately outside this explicitly selected scope.
    (roots["demo"] / "README.md").write_text("not verified")
    selected = {name: roots[name] for name in ("model", "dataset")}
    report = CLI.verify_release(trusted, selected, tmp_path / "verified.json", skip_demo=True)
    assert report["status"] == "verified"
    assert set(report["bundles"]) == {"model", "dataset"}
    assert report["skipped_bundles"] == ["demo"]
    assert "demo explicitly skipped and not checked" in report["scope"]
    assert report["dataset_counts"] == counts
    for name in selected:
        assert report["bundles"][name]["release_files_sha256"] == CLI.sha256(
            selected[name] / "release-files.json"
        )


@pytest.mark.parametrize("bundle", ["model", "dataset"])
def test_skip_demo_keeps_payload_integrity_checks(release, tmp_path, bundle):
    trusted, roots, _ = release
    path = roots[bundle] / "README.md"
    path.write_bytes(b"X" * path.stat().st_size)
    selected = {name: roots[name] for name in ("model", "dataset")}
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        CLI.verify_release(trusted, selected, tmp_path / "verified.json", skip_demo=True)


@pytest.mark.parametrize(
    "names,skip_demo",
    [(("model", "dataset"), False), (("model", "dataset", "demo"), True), (("model",), True)],
)
def test_roots_must_match_explicit_verification_scope(release, tmp_path, names, skip_demo):
    trusted, roots, _ = release
    selected = {name: roots[name] for name in names}
    with pytest.raises(ValueError, match="verification needs"):
        CLI.verify_release(trusted, selected, tmp_path / "verified.json", skip_demo=skip_demo)


@pytest.mark.parametrize("flags", [[], ["--skip-demo", "--demo", "unused"]])
def test_cli_requires_exactly_one_demo_mode(release, tmp_path, flags):
    trusted, roots, _ = release
    with pytest.raises(SystemExit) as error:
        CLI.main(
            [
                "--trusted-manifest",
                str(trusted),
                "--model",
                str(roots["model"]),
                "--dataset",
                str(roots["dataset"]),
                "--output",
                str(tmp_path / "verified.json"),
                *flags,
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "verified.json").exists()


def test_cli_skip_demo_records_scope(release, tmp_path, capsys):
    trusted, roots, _ = release
    output = tmp_path / "verified.json"
    assert (
        CLI.main(
            [
                "--trusted-manifest",
                str(trusted),
                "--model",
                str(roots["model"]),
                "--dataset",
                str(roots["dataset"]),
                "--skip-demo",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["skipped_bundles"] == ["demo"]
    assert json.loads(capsys.readouterr().out)["skipped_bundles"] == ["demo"]
