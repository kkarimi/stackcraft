"""Reproducible imitation data with episode splits and observation deduplication."""

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stackcraft.engine import legal_actions, new_game, step
from stackcraft.expert import SearchExpert
from stackcraft.players import HeuristicPlayer, Player, RandomPlayer, observe
from stackcraft.schema import RULES_VERSION, GameState

SCHEMA_VERSION = 1
DEVELOPMENT_SEEDS = frozenset((*range(20), 1000))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "engine.py",
            "pieces.py",
            "schema.py",
            "players/__init__.py",
            "expert.py",
            "data.py",
        )
    }


def observation_hash(observation: dict[str, Any]) -> str:
    """Colors do not change dynamics. Keep all visible geometry and options."""
    normalized = dict(observation)
    normalized["board"] = [[int(cell != 0) for cell in row] for row in observation["board"]]
    return digest(normalized)


@dataclass(frozen=True)
class DatasetConfig:
    train_seeds: tuple[int, ...] = tuple(range(10000, 10024))
    validation_seeds: tuple[int, ...] = tuple(range(20000, 20006))
    reserved_test_seeds: tuple[int, ...] = tuple(range(30000, 30200))
    max_pieces: int = 40
    behavior_cycle: tuple[str, ...] = ("random", "heuristic", "expert")

    def __post_init__(self) -> None:
        pools = (self.train_seeds, self.validation_seeds, self.reserved_test_seeds)
        all_seeds = [seed for pool in pools for seed in pool]
        if any(not pool for pool in pools):
            raise ValueError("all seed pools must be nonempty")
        if any(type(seed) is not int for seed in all_seeds):
            raise ValueError("seeds must be integers")
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError("seed pools must be unique and disjoint")
        if set(all_seeds) & DEVELOPMENT_SEEDS:
            raise ValueError("development seeds must not enter dataset pools")
        if type(self.max_pieces) is not int or self.max_pieces < 1:
            raise ValueError("max_pieces must be a positive integer")
        if not self.behavior_cycle or any(
            name not in ("random", "heuristic", "expert") for name in self.behavior_cycle
        ):
            raise ValueError("behavior_cycle must contain random, heuristic, or expert")


@dataclass(frozen=True)
class DatasetBundle:
    records: dict[str, list[dict[str, Any]]]
    manifest: dict[str, Any]


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(canonical_json(record) + "\n" for record in records)


def generate_dataset(config: DatasetConfig, source_commit: str) -> DatasetBundle:
    """Training owns duplicates before validation; conflicting labels are fatal."""
    base_commit = source_commit.removesuffix("+working-tree")
    if len(base_commit) != 40 or any(c not in "0123456789abcdef" for c in base_commit):
        raise ValueError("source_commit must be a full Git hash, optionally +working-tree")
    teacher = SearchExpert()
    records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    seen: dict[str, tuple[str, str, dict[str, int]]] = {}
    excluded = {split: {"within_split": 0, "cross_split": 0} for split in records}
    collected = dict.fromkeys(records, 0)
    for split, seeds in (("train", config.train_seeds), ("validation", config.validation_seeds)):
        for episode_index, seed in enumerate(seeds):
            name = config.behavior_cycle[episode_index % len(config.behavior_cycle)]
            behavior: Player = (
                RandomPlayer(seed + 1_000_000)
                if name == "random"
                else HeuristicPlayer()
                if name == "heuristic"
                else teacher
            )
            state = new_game(seed)
            while not state.terminal and state.piece_index < config.max_pieces:
                observation = observe(state)
                values = teacher.action_values(observation)
                action_id = max(values, key=lambda action: values[action])
                raw_observation = json.loads(canonical_json(asdict(observation)))
                key = observation_hash(raw_observation)
                collected[split] += 1
                if key in seen:
                    old_split, old_action, old_values = seen[key]
                    if old_action != action_id or old_values != values:
                        raise ValueError("duplicate observation has conflicting teacher labels")
                    excluded[split]["within_split" if old_split == split else "cross_split"] += 1
                else:
                    seen[key] = (split, action_id, values)
                    episode_id = f"seed-{seed}"
                    records[split].append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "rules_version": RULES_VERSION,
                            "id": f"{episode_id}-turn-{state.piece_index}",
                            "split": split,
                            "episode_id": episode_id,
                            "seed": seed,
                            "turn": state.piece_index,
                            "behavior": {"name": behavior.name, "revision": behavior.revision},
                            "observation": raw_observation,
                            "observation_hash": key,
                            "action_id": action_id,
                            "action_values": values,
                            "teacher_revision": teacher.revision,
                            "source_commit": source_commit,
                        }
                    )
                chosen = action_id if name == "expert" else behavior.choose(observation).action_id
                state = step(state, chosen).state
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rules_version": RULES_VERSION,
        "source_commit": source_commit,
        "source_hashes": source_hashes(),
        "source_note": "File hashes identify source; commit may precede working-tree edits.",
        "teacher_revision": teacher.revision,
        "teacher": {"preview_pieces": 1, "search": "exhaustive legal placements", "optimal": False},
        "config": json.loads(canonical_json(asdict(config))),
        "seed_pools": {
            "train": list(config.train_seeds),
            "validation": list(config.validation_seeds),
            "reserved_test": list(config.reserved_test_seeds),
        },
        "test_trajectories_generated": False,
        "deduplication": {"normalization": "board occupancy", "exclusions": excluded},
        "splits": {
            split: {
                "collected": collected[split],
                "records": len(rows),
                "sha256": hashlib.sha256(_jsonl(rows).encode()).hexdigest(),
                "behavior_records": dict(
                    sorted(Counter(row["behavior"]["name"] for row in rows).items())
                ),
            }
            for split, rows in records.items()
        },
    }
    manifest["config_sha256"] = digest(manifest["config"])
    audit_dataset(records, manifest)
    return DatasetBundle(records, manifest)


def audit_dataset(
    records: dict[str, list[dict[str, Any]]], manifest: dict[str, Any]
) -> dict[str, int]:
    """Check provenance, legal labels, stable tie-breaking, and split leakage."""
    if set(records) != {"train", "validation"}:
        raise ValueError("dataset must contain train and validation only")
    if manifest["test_trajectories_generated"] is not False:
        raise ValueError("test trajectories must remain reserved")
    config = DatasetConfig(**manifest["config"])
    if manifest["config_sha256"] != digest(manifest["config"]):
        raise ValueError("configuration hash mismatch")
    expected_pools = {
        "train": list(config.train_seeds),
        "validation": list(config.validation_seeds),
        "reserved_test": list(config.reserved_test_seeds),
    }
    if manifest["seed_pools"] != expected_pools:
        raise ValueError("seed pool manifest differs from configuration")
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    episode_splits: dict[str, str] = {}
    for split, rows in records.items():
        if manifest["splits"][split]["sha256"] != hashlib.sha256(_jsonl(rows).encode()).hexdigest():
            raise ValueError("record content hash mismatch")
        if manifest["splits"][split]["records"] != len(rows):
            raise ValueError("record count mismatch")
        for row in rows:
            if row["split"] != split or row["seed"] not in expected_pools[split]:
                raise ValueError("record is assigned to the wrong split")
            if row["episode_id"] != f"seed-{row['seed']}":
                raise ValueError("episode identifier differs from seed")
            if row["id"] != f"{row['episode_id']}-turn-{row['turn']}" or row["id"] in seen_ids:
                raise ValueError("invalid or duplicate record ID")
            if type(row["turn"]) is not int or not 0 <= row["turn"] < config.max_pieces:
                raise ValueError("turn exceeds episode limits")
            seen_ids.add(row["id"])
            if episode_splits.setdefault(row["episode_id"], split) != split:
                raise ValueError("episode overlaps dataset splits")
            for field in ("schema_version", "rules_version", "source_commit", "teacher_revision"):
                if row[field] != manifest[field]:
                    raise ValueError(f"record {field} differs from manifest")
            observation = row["observation"]
            if set(observation) != {
                "board",
                "current",
                "next_piece",
                "legal_actions",
                "rules_version",
            }:
                raise ValueError("observation contains unexpected or hidden fields")
            state = GameState(
                tuple(tuple(r) for r in observation["board"]),
                0,
                0,
                observation["current"],
                observation["next_piece"],
            )
            if observation != json.loads(canonical_json(asdict(observe(state)))):
                raise ValueError("observation contains incorrect legal actions or rules")
            key = observation_hash(observation)
            if row["observation_hash"] != key or key in seen_hashes:
                raise ValueError("observation hash mismatch or duplicate across dataset")
            seen_hashes.add(key)
            values = row["action_values"]
            actions = legal_actions(state)
            if (
                not actions
                or set(values) != {a.id for a in actions}
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in values.values()
                )
            ):
                raise ValueError("teacher values must be finite and cover legal actions")
            if row["action_id"] != max(actions, key=lambda a: values[a.id]).id:
                raise ValueError("teacher label differs from action values or tie rule")
    return {split: len(rows) for split, rows in records.items()}


def write_dataset(bundle: DatasetBundle, directory: Path) -> None:
    """Write a verified immutable artifact directory; refuse existing files."""
    audit_dataset(bundle.records, bundle.manifest)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{split}.jsonl" for split in bundle.records]
    paths.append(directory / "manifest.json")
    if any(path.exists() for path in paths):
        raise FileExistsError("dataset artifact files already exist")
    for split, rows in bundle.records.items():
        (directory / f"{split}.jsonl").write_text(_jsonl(rows))
    (directory / "manifest.json").write_text(canonical_json(bundle.manifest) + "\n")
