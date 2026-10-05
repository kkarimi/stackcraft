import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stackcraft.provenance import file_sha256, source_identity


def write_manifest(root: Path, *, commit: str = "a" * 40, dirty: bool = False) -> None:
    hashes = {
        str(path.relative_to(root)): file_sha256(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "source-manifest.json"
    }
    (root / "source-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_commit": commit,
                "source_dirty": dirty,
                "source_hashes": hashes,
            }
        )
    )


def initialize_git(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / "README.md").write_text("test")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )


def test_exact_git_root_identity_and_dirty_state(tmp_path):
    initialize_git(tmp_path)
    identity = source_identity(tmp_path)
    assert identity["source_identity_method"] == "exact-git-root"
    assert identity["source_dirty"] is False
    assert len(identity["source_commit"]) == 40
    (tmp_path / "README.md").write_text("changed")
    assert source_identity(tmp_path)["source_dirty"] is True


def test_unrelated_parent_repository_is_never_claimed(tmp_path):
    initialize_git(tmp_path)
    child = tmp_path / "downloaded-code"
    child.mkdir()
    with pytest.raises(ValueError, match="no exact Stackcraft"):
        source_identity(child)
    (child / "README.md").write_text("downloaded source")
    write_manifest(child)
    identity = source_identity(child)
    assert identity["source_identity_method"] == "published-source-manifest"
    assert identity["source_commit"] == "a" * 40
    assert identity["source_dirty"] is False


def test_published_source_changes_missing_files_and_new_modules_are_marked(tmp_path):
    package = tmp_path / "src/stackcraft"
    package.mkdir(parents=True)
    (package / "a.py").write_text("a = 1")
    (package / "b.py").write_text("b = 1")
    write_manifest(tmp_path)
    assert source_identity(tmp_path)["source_dirty"] is False
    (package / "a.py").write_text("a = 2")
    (package / "b.py").unlink()
    (package / "c.py").write_text("c = 1")
    identity = source_identity(tmp_path)
    assert identity["source_dirty"] is True
    assert identity["modified_source_files"] == [
        "src/stackcraft/a.py",
        "src/stackcraft/b.py",
        "src/stackcraft/c.py",
    ]


@pytest.mark.parametrize(
    "field,value",
    [("source_commit", "unversioned"), ("schema_version", True), ("source_dirty", "false")],
)
def test_malformed_published_identity_is_rejected(tmp_path, field, value):
    (tmp_path / "README.md").write_text("source")
    write_manifest(tmp_path)
    path = tmp_path / "source-manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid published"):
        source_identity(tmp_path)


def test_public_source_download_cpu_cli_smoke_without_git(tmp_path):
    """Use copied source via PYTHONPATH: no Git ancestry and no model dependency."""
    project = Path(__file__).resolve().parents[1]
    package = tmp_path / "src/stackcraft"
    shutil.copytree(
        project / "src/stackcraft", package, ignore=shutil.ignore_patterns("__pycache__")
    )
    (tmp_path / "scripts").mkdir()
    for name in ("train_clef.py", "evaluate_clef.py"):
        shutil.copyfile(project / "scripts" / name, tmp_path / "scripts" / name)
    shutil.copyfile(project / "uv.lock", tmp_path / "uv.lock")
    write_manifest(tmp_path)
    environment = {**os.environ, "PYTHONPATH": str(tmp_path / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stackcraft.cli",
            "tournament",
            "--seed-start",
            "7",
            "--episodes",
            "1",
            "--max-pieces",
            "2",
            "--output",
            str(tmp_path / "offline.json"),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Saved 1 paired development episodes" in result.stdout
    script = """
from pathlib import Path
import importlib.util
import json
import sys
from stackcraft import cli, data
from stackcraft.provenance import source_identity
from stackcraft.data import DatasetConfig, generate_dataset
identity = source_identity(Path.cwd())
assert identity['source_identity_method'] == 'published-source-manifest'
assert not identity['source_dirty']
config = DatasetConfig(
    train_seeds=(10000,), validation_seeds=(20000,), reserved_test_seeds=(40000,), max_pieces=1
)
bundle = generate_dataset(config, source_commit=identity['source_commit'])
assert bundle.manifest['source_commit'] == 'a' * 40
assert bundle.records['train']
data.DatasetConfig = lambda **kwargs: DatasetConfig(**kwargs) if kwargs else config
sys.argv = ['stackcraft', 'generate-data', '--output', 'tiny-cli-data']
cli.main()
assert json.loads(Path('tiny-cli-data/manifest.json').read_text())['source_commit'] == 'a' * 40
for name, function in [('train_clef.py', 'source_metadata'), ('evaluate_clef.py', '_provenance')]:
    path = Path('scripts') / name
    spec = importlib.util.spec_from_file_location('public_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = getattr(module, function)()
    assert metadata['source_identity_method'] == 'published-source-manifest'
    assert metadata['source_commit'] == 'a' * 40
    assert metadata['source_dirty'] is False
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=environment, check=True)
