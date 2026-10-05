"""Training orchestration checks; no GPU or pretrained weights are loaded."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def script():
    path = Path(__file__).resolve().parents[1] / "scripts/train_clef.py"
    spec = importlib.util.spec_from_file_location("stackcraft_test_train_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_epoch_covers_every_training_row_and_correct_partial_group(script):
    first = script.accumulation_groups(827, 8, epoch=1)
    assert len(first) == 104
    assert len(first[-1]) == 3
    assert sorted(index for group in first for index in group) == list(range(827))
    assert first == script.accumulation_groups(827, 8, epoch=1)
    assert first != script.accumulation_groups(827, 8, epoch=2)


@pytest.mark.parametrize(
    "arguments", [["--epochs", "3"], ["--learning-rate", "nan"], ["--accumulation", "0"]]
)
def test_invalid_configuration_fails_before_any_model_load(script, tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        script.main(["--output", str(tmp_path / "new"), *arguments])
    assert error.value.code == 2
    assert not (tmp_path / "new").exists()


def test_existing_output_is_never_overwritten(script, tmp_path):
    with pytest.raises(SystemExit) as error:
        script.main(["--output", str(tmp_path)])
    assert error.value.code == 2


def test_only_fixed_train_and_validation_are_read_and_audited(script, tmp_path, monkeypatch):
    train = [{"id": "train-a"}, {"id": "train-b"}]
    validation = [{"id": "validation-a"}]
    hashes = {}
    for split, rows in (("train", train), ("validation", validation)):
        content = "".join(json.dumps(row) + "\n" for row in rows)
        (tmp_path / f"{split}.jsonl").write_text(content)
        hashes[split] = hashlib.sha256(content.encode()).hexdigest()
    (tmp_path / "test.jsonl").write_text("this file must never be opened as JSON")
    manifest = {"source_commit": "test-source", "config_sha256": "test-config"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(script, "STUDY_HASHES", hashes)
    monkeypatch.setattr(script, "STUDY_COUNTS", {"train": 2, "validation": 1})
    calls = []
    monkeypatch.setattr(script, "audit_dataset", lambda records, manifest: calls.append(records))
    loaded, metadata = script.load_study(tmp_path)
    assert loaded == train
    assert calls == [{"train": train, "validation": validation}]
    assert metadata["test_trajectories_used"] is False
    assert metadata["validation_used_for_training"] is False
    (tmp_path / "train.jsonl").write_text('{"id":"tampered"}\n')
    with pytest.raises(ValueError, match="frozen study-v1 SHA256"):
        script.load_study(tmp_path)


def test_native_loss_accumulation_matches_actual_group_mean(script, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from stackcraft.training import decision_loss

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(2))
            self.language_model = torch.nn.Identity()

        def forward(self, batch):
            return [[self.logits]]

    encoded = SimpleNamespace(
        input_ids=(1, 2),
        questions=(SimpleNamespace(question_type=1, option_ids=("r0x0", "r0x1")),),
    )
    monkeypatch.setattr(script, "row_observation", lambda row: row)
    monkeypatch.setattr(script, "encode_observation", lambda *args: encoded)
    model = TinyModel()
    reference = TinyModel()
    rows = [{"id": f"row-{index}", "action_id": "r0x1"} for index in range(5)]
    player = SimpleNamespace(
        model=model,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        native=SimpleNamespace(collate_records=lambda *args: {}),
        max_length=4096,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    for _ in range(2):
        expected_optimizer.zero_grad(set_to_none=True)
        decision_loss(reference.logits, encoded, "r0x1").backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
        expected_optimizer.step()
    outcome = script.train_epoch(
        player, rows, optimizer, epoch=1, accumulation=3, mode="head", output=tmp_path
    )
    torch.testing.assert_close(model.logits, reference.logits, rtol=0, atol=1e-8)
    assert outcome["examples"] == 5
    assert outcome["optimizer_steps"] == 2
    events = [
        json.loads(line) for line in (tmp_path / "epoch-01-events.jsonl").read_text().splitlines()
    ]
    microsteps = [event for event in events if event["event"] == "microstep"]
    assert {event["row_id"] for event in microsteps} == {row["id"] for row in rows}
    assert [event["accumulation_group_size"] for event in microsteps] == [3, 3, 3, 2, 2]
    assert all(event["peak_allocated_bytes"] == 0 for event in microsteps)
