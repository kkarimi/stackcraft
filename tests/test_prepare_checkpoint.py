"""Preparation only copies bytes and edits documentation; it never loads a model."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def preparer():
    path = Path(__file__).parents[1] / "scripts/prepare_checkpoint.py"
    spec = importlib.util.spec_from_file_location("stackcraft_prepare_checkpoint_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw(preparer, tmp_path):
    source = tmp_path / "raw" / "epoch-01"
    (source / "adapter").mkdir(parents=True)
    for name in preparer.REQUIRED_FILES:
        (source / name).write_bytes(b"nonempty-test-weight-bytes")
    (source / "training_config.json").write_text(
        json.dumps(
            {
                "base_model": preparer.MODEL_ID,
                "base_revision": preparer.MODEL_REVISION,
                "mode": "lora",
            }
        )
    )
    (source / "adapter/adapter_config.json").write_text('{"r":4}')
    (source / "reference.json").write_text('{"probabilities": [{"r0x0":1}]}')
    (source / "adapter/README.md").write_text("placeholder /private/cache/path")
    (source / "additional.bin").write_bytes(b"preserve extra file too")
    return source


def test_only_adapter_card_changes_and_provenance_is_outside_candidate(preparer, raw, tmp_path):
    output = tmp_path / "candidates" / "epoch-01"
    original = preparer.inventory(raw)
    report = preparer.prepare(raw, output)
    assert report["changed_files"] == ["adapter/README.md"]
    assert preparer.inventory(raw) == original == report["source_sha256"]
    assert preparer.inventory(output) == report["output_sha256"]
    for name, digest in original.items():
        if name != "adapter/README.md":
            assert report["output_sha256"][name] == digest
    text = (output / "adapter/README.md").read_text()
    assert preparer.MODEL_ID in text and preparer.MODEL_REVISION in text
    assert "joint_head.safetensors" in text and "load_checkpoint" in text
    assert "/private/cache" not in text and "placeholder" not in text
    sidecar = output.parent / "epoch-01-preparation.json"
    assert json.loads(sidecar.read_text()) == report
    assert set(preparer.inventory(output)) == set(original)


@pytest.mark.parametrize("existing", ["output", "provenance"])
def test_refuses_existing_destination_or_provenance(preparer, raw, tmp_path, existing):
    output = tmp_path / "candidates" / "epoch-01"
    output.parent.mkdir()
    if existing == "output":
        output.mkdir()
        sentinel = output / "user-file"
    else:
        sentinel = output.parent / "epoch-01-preparation.json"
    sentinel.write_text("keep this")
    with pytest.raises(FileExistsError):
        preparer.prepare(raw, output)
    assert sentinel.read_text() == "keep this"


@pytest.mark.parametrize(
    "location", ["source-root", "source-file", "source-directory", "output-parent"]
)
def test_rejects_symlinks(preparer, raw, tmp_path, location):
    source = raw
    output = tmp_path / "candidates" / "epoch-01"
    if location == "source-root":
        source = tmp_path / "source-link"
        source.symlink_to(raw, target_is_directory=True)
    elif location == "source-file":
        (raw / "linked-file").symlink_to(raw / "reference.json")
    elif location == "source-directory":
        (raw / "linked-directory").symlink_to(raw / "adapter", target_is_directory=True)
    else:
        actual = tmp_path / "actual-output-parent"
        actual.mkdir()
        output.parent.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        preparer.prepare(source, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "missing", ["reference.json", "joint_head.safetensors", "adapter/adapter_model.safetensors"]
)
def test_incomplete_checkpoint_is_rejected(preparer, raw, tmp_path, missing):
    (raw / missing).unlink()
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="missing"):
        preparer.prepare(raw, output)
    assert not output.exists()


def test_nested_output_cannot_modify_raw_checkpoint(preparer, raw):
    original = preparer.inventory(raw)
    with pytest.raises(ValueError, match="nested"):
        preparer.prepare(raw, raw / "candidate")
    assert preparer.inventory(raw) == original


def test_copy_corruption_is_detected_and_owned_partial_output_removed(
    preparer, raw, tmp_path, monkeypatch
):
    output = tmp_path / "candidate"
    original = preparer.inventory(raw)
    copy2 = preparer.shutil.copy2

    def corrupt(source, destination, **kwargs):
        result = copy2(source, destination, **kwargs)
        if destination.name == "joint_head.safetensors":
            destination.write_bytes(b"corrupted")
        return result

    monkeypatch.setattr(preparer.shutil, "copy2", corrupt)
    with pytest.raises(RuntimeError, match="other than"):
        preparer.prepare(raw, output)
    assert not output.exists()
    assert not (output.parent / "candidate-preparation.json").exists()
    assert preparer.inventory(raw) == original
