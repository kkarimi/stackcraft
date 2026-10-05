import copy
import hashlib
import json

import pytest

from stackcraft.data import (
    DatasetConfig,
    audit_dataset,
    canonical_json,
    generate_dataset,
    observation_hash,
    write_dataset,
)

COMMIT = "412be98cf90bc7e3e60cd39d8bf9d479bcf65d27"


@pytest.fixture
def bundle():
    return generate_dataset(
        DatasetConfig(
            train_seeds=(10000, 10001, 10002), validation_seeds=(20000, 20001, 20002), max_pieces=2
        ),
        COMMIT,
    )


def refresh_hash(records, manifest, split):
    data = "".join(canonical_json(row) + "\n" for row in records[split])
    manifest["splits"][split]["sha256"] = hashlib.sha256(data.encode()).hexdigest()
    manifest["splits"][split]["records"] = len(records[split])


def test_dataset_is_reproducible_and_never_collects_test_episodes(bundle) -> None:
    assert bundle == generate_dataset(DatasetConfig(**bundle.manifest["config"]), COMMIT)
    assert audit_dataset(bundle.records, bundle.manifest) == {
        split: len(rows) for split, rows in bundle.records.items()
    }
    assert bundle.manifest["test_trajectories_generated"] is False
    assert bundle.manifest["seed_pools"]["reserved_test"] == list(range(30000, 30200))
    assert all(
        row["seed"] not in range(30000, 30200) for rows in bundle.records.values() for row in rows
    )
    assert set(bundle.manifest["source_hashes"]) == {
        "engine.py",
        "pieces.py",
        "schema.py",
        "players/__init__.py",
        "expert.py",
        "data.py",
    }


def test_observation_hash_ignores_colors_not_geometry(bundle) -> None:
    observation = copy.deepcopy(bundle.records["train"][1]["observation"])
    colored = copy.deepcopy(observation)
    colored["board"] = [[7 if cell else 0 for cell in row] for row in colored["board"]]
    assert observation_hash(observation) == observation_hash(colored)
    colored["board"][0][0] = 1
    assert observation_hash(observation) != observation_hash(colored)


def test_collector_deduplicates_identical_starting_observations() -> None:
    # At least two of 60 starts share one of the 49 possible current/preview pairs.
    result = generate_dataset(
        DatasetConfig(
            train_seeds=tuple(range(10000, 10060)), validation_seeds=(20000,), max_pieces=1
        ),
        COMMIT,
    )
    assert result.manifest["deduplication"]["exclusions"]["train"]["within_split"] > 0
    hashes = [row["observation_hash"] for rows in result.records.values() for row in rows]
    assert len(hashes) == len(set(hashes))


@pytest.mark.parametrize(
    "field,value", [("action_id", "not-legal"), ("seed", 30000), ("source_commit", "bad")]
)
def test_audit_rejects_invalid_provenance_or_labels_even_after_rehash(bundle, field, value) -> None:
    rows, manifest = copy.deepcopy(bundle.records), copy.deepcopy(bundle.manifest)
    rows["train"][0][field] = value
    refresh_hash(rows, manifest, "train")
    with pytest.raises(ValueError):
        audit_dataset(rows, manifest)


def test_audit_rejects_hidden_fields(bundle) -> None:
    rows, manifest = copy.deepcopy(bundle.records), copy.deepcopy(bundle.manifest)
    rows["train"][0]["observation"]["seed"] = 10000
    refresh_hash(rows, manifest, "train")
    with pytest.raises(ValueError, match="hidden fields"):
        audit_dataset(rows, manifest)


def test_audit_detects_cross_split_identical_observation(bundle) -> None:
    rows, manifest = copy.deepcopy(bundle.records), copy.deepcopy(bundle.manifest)
    source = copy.deepcopy(rows["train"][0])
    target = rows["validation"][0]
    for field in ("observation", "observation_hash", "action_values", "action_id"):
        target[field] = source[field]
    refresh_hash(rows, manifest, "validation")
    with pytest.raises(ValueError, match="duplicate"):
        audit_dataset(rows, manifest)


def test_written_hashes_match_and_artifacts_are_not_overwritten(bundle, tmp_path) -> None:
    write_dataset(bundle, tmp_path)
    for split in bundle.records:
        assert (
            hashlib.sha256((tmp_path / f"{split}.jsonl").read_bytes()).hexdigest()
            == bundle.manifest["splits"][split]["sha256"]
        )
    assert json.loads((tmp_path / "manifest.json").read_text()) == bundle.manifest
    with pytest.raises(FileExistsError):
        write_dataset(bundle, tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_seeds": (0,)},
        {"train_seeds": (1000,)},
        {"validation_seeds": (10000,)},
        {"train_seeds": (True,)},
        {"max_pieces": 0},
        {"behavior_cycle": ("unknown",)},
    ],
)
def test_dataset_pool_and_config_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        DatasetConfig(**kwargs)


def test_working_tree_provenance_is_explicit() -> None:
    result = generate_dataset(
        DatasetConfig(train_seeds=(10000,), validation_seeds=(20000,), max_pieces=1),
        COMMIT + "+working-tree",
    )
    assert result.manifest["source_commit"] == COMMIT + "+working-tree"
    assert result.manifest["source_hashes"]["data.py"]
