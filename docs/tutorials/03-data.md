# Tutorial 03 — Teach from search without leaking future pieces

The dataset records recommendations from a bounded search expert. These are
synthetic labels from a specified algorithm, not human labels or globally optimal
Tetris moves. The expert sees the current piece and one preview, exactly as other
players do.

## Write the experiment before collecting examples

Starting commit: `412be98cf90bc7e3e60cd39d8bf9d479bcf65d27`, plus the expert/data
implementation committed at integration. Preregistered pilot: training seeds
10000–10023, validation seeds 20000–20005, at most 40 pieces per episode. Cycle
behavior policies random, heuristic, expert by episode index within each split.
Label every visited nonterminal state with the expert. Generate the pilot twice
and require identical records, manifest and JSONL hashes. Stop each job at 120
seconds or the first audit failure. Output the first bundle under `data/pilot-v1`.

Reserve seeds 30000–30199 for final evaluation; generate no trajectories from
them during dataset work. Development seeds 0–19 and 1000 are excluded from all
dataset pools. A separate development comparison uses only seeds 0–4 and a
100-piece cap, with heuristic and expert. Its purpose is to inspect search behavior
and cost, not measure neural-model improvement. Do not tune weights using these
results. Expected comparison artifact: `runs/expert-development.json`.

## Search only information available to the player

For each legal current placement, the expert clears completed rows and enumerates
all legal placements of the visible preview. It scores:

    8 * current_lines + max(preview_afterboard_value)

The afterboard value is the same untuned `-height -4*holes -bumpiness +8*lines`
heuristic from Tutorial 02. If the preview has no legal placement, its value is
-10000. This is a two-placement search with no pruning. Equal values select the
first engine action in rotation/column order.

The search constructs a synthetic state with dummy seed/index only to enumerate
geometry. It calls `place`, never `step`, so it cannot query the piece after the
preview. Tests make sequence lookup fail during search to enforce that boundary.
Searching more known future pieces would give the teacher information unavailable
to the model; it would solve a different task. Long-horizon search is also more
expensive and is unnecessary for this first imitation-learning experiment.

## Collect recovery states as well as expert play

Only collecting expert trajectories produces many tidy boards and few recovery
situations. Cycling random, greedy heuristic, and expert behavior visits different
board conditions. Every behavior's state receives the same expert label. The
random policy has its own RNG, seeded with episode seed plus one million; this
controls collection reproducibility without exposing any seed to the teacher.

Each row retains the raw visible observation, action values for every legal move,
chosen move, behavior policy, episode/turn, rules, teacher revision and source
commit. Provenance fields are for auditing; training inputs must use only
`observation`. In particular, never serialize the whole dataset row as model state.

## Split episodes first, then remove duplicate observations

Seed pools are disjoint before generation. All retained states from one episode
remain in one split. Exact visible observations can still repeat across episodes,
especially empty boards. Deduplication hashes the full observation after replacing
board colors with occupied/empty cells, because colors do not affect game dynamics.
Current piece, preview, legal geometry, and rules remain in that hash.

Process training before validation and keep the first unique observation. Remove
later duplicates within either split and across splits, recording both counts.
Require duplicate teacher labels and action values to agree; a conflict stops the
run. This deterministic policy sacrifices some validation rows rather than letting
identical positions inflate reported generalization. Full episode IDs and seed
reservations are checked separately from observation hashes.

## Reproduce and audit

```python
from pathlib import Path
from stackcraft.data import DatasetConfig, audit_dataset, generate_dataset, write_dataset
from stackcraft.provenance import source_identity

config = DatasetConfig()
identity = source_identity(Path.cwd())
commit = identity["source_commit"] + ("+working-tree" if identity["source_dirty"] else "")
output = Path("data/pilot-v1")
assert not output.exists(), "Choose a new output directory"
bundle = generate_dataset(config, source_commit=commit)
assert bundle == generate_dataset(config, source_commit=bundle.manifest["source_commit"])
print(audit_dataset(bundle.records, bundle.manifest))
write_dataset(bundle, output)
```

Run with `uv run --locked python` from the repository root or the public bundle's
`code/` directory. Source identity uses that exact Git root or verifies the bundled
source manifest, so private Git access is not required. This creates a new pilot
with the current source identity, not necessarily the original study's bytes.
The writer refuses existing artifact files so a new experiment cannot silently
replace recorded evidence.
The manifest stores the complete configuration, seed reservations, teacher formula
revision, collection and exclusion counts, and SHA-256 hashes of canonical JSONL.
Auditing reconstructs legal placements and verifies label argmax with the tie rule,
finite values, provenance, duplicate exclusion, and absence of hidden observation
fields. It does not prove that the teacher is optimal. Evaluation must still test
complete games, not just how accurately a model copies these labels.

The manifest also fingerprints `engine.py`, `pieces.py`, `schema.py`,
`players/__init__.py`, `expert.py`, and `data.py`. During development, append
`+working-tree` to the commit identifier when source edits are not committed.
Before study training, freeze the dataset against its committed generator source.
Preserve those exact bytes through publication; later documentation or packaging
commits are not reasons to regenerate the training dataset. A source hash records
what ran; it must not imply that an earlier commit includes later code.

## Pilot observations

The preregistered 2026-10-05 pilot collected 833 training and 217 validation
positions. Six repeated training positions and two validation positions already
present in training were removed, leaving **827 training / 215 validation** rows.
Both complete generation runs produced identical bundles. Each took about 9.05
seconds; a separate structural audit took 0.49 seconds on the development machine.

The historical pilot's JSONL hashes are:

- Train: `bccf62673fb3160ac957650d400edcd6efb28e56b5e212940aba3281fb4bb82d`.
- Validation: `e6e87cad6aea503ef41894d7aa53cd80d635d540c7d2f2942f79a033692cce76`.

These artifacts used the original pilot source recorded in their manifest. A later
change allowed the explicit `+working-tree` commit suffix; that change does not
retroactively change which code produced this pilot. A final committed regeneration
will have different provenance hashes even when game examples remain the same.

For an additional label check, an independently written nested loop recomputed
every action value on 16 positions: first and highest-stack examples from each of
the six split/behavior groups, then further highest-stack positions to reach 16.
The sample included height-20 recovery boards. All values and chosen labels
matched. Evidence is `runs/data-pilot-independent-audit.json`. This checks the
implementation against the declared search formula, not optimality of that formula.

The five-seed development tournament produced mean lines 37.8 for the heuristic
and 38.6 for search; both reached the 100-piece cap in all five games with no
errors. Mean search decision time was 8.33 ms versus 0.183 ms for the heuristic.
The collection and comparison jobs ran concurrently on CPU, so these times are
illustrative, not isolated performance benchmarks. Five capped games are insufficient
to establish broad superiority. No neural model was evaluated.

## Reproduce from the public source bundle without private Git access

The model release includes `code/` with source, a lockfile and a source manifest.
From that directory, `uv sync --locked` installs the game/data tools. The regular
`stackcraft generate-data` command records the bundle's source identity and flags
modified source files. Its provenance will therefore differ from the original
M3 commit even when generator logic is unchanged.

The probe and full-training commands in later tutorials use `data/study-v1`.
For a fresh reproduction, the command below creates that directory with the exact
original study bytes. Alternatively, place the released dataset's `train.jsonl`,
`validation.jsonl` and `manifest.json` in that directory and verify them using the
release audit. Do not substitute the newly generated pilot above for the study.

To reproduce the original study bytes exactly, first verify that every generator
source file matches the original dataset manifest, then use its recorded commit
in the generator API. This avoids falsely stamping changed code with an old commit:

```bash
uv run --locked python - <<'PY'
import hashlib
import json
from pathlib import Path
from stackcraft.data import DatasetConfig, generate_dataset, write_dataset

manifest = json.loads(Path('reports/dataset-manifest.json').read_text())
for name, expected in manifest['source_hashes'].items():
    path = Path('src/stackcraft') / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
output = Path('data/study-v1')
assert not output.exists(), 'Choose a new output directory'
bundle = generate_dataset(DatasetConfig(), source_commit=manifest['source_commit'])
assert bundle.manifest == manifest, 'Generated manifest differs from the frozen study'
write_dataset(bundle, output)
for split in ('train', 'validation'):
    actual = hashlib.sha256((output / f'{split}.jsonl').read_bytes()).hexdigest()
    assert actual == manifest['splits'][split]['sha256'], split
print('Both original split hashes match')
PY
```

If `data/study-v1` already exists, retain and audit it rather than overwrite it.
For a separate regeneration, choose a new output directory and supply that path
with `--dataset` in subsequent probe, training and evaluation commands. Use only
the audited downloaded dataset or the exact hash-matching reconstruction to
reproduce study training. If any source hash fails, investigate or generate a
clearly new dataset; do not suppress the check or reuse the original study's identity.
