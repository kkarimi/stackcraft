# Stackcraft

A playable, deterministic falling-block game for studying decision models. The goal is a fair comparison of a heuristic player, unchanged Clef-flash, and a fine-tuned model on identical piece sequences.

**Status:** the playable game, data, native Clef baseline and real GPU training/reload gates are complete. The full two-epoch study is running; validation selection, held-out results and publication remain pending. See [the project plan](plan.md) for milestones and acceptance criteria.

## Run locally

Install [uv](https://docs.astral.sh/uv/), then from this directory:

```bash
uv sync --locked
uv run --locked stackcraft serve --port 8087
```

Open http://127.0.0.1:8087. The game needs no GPU or model credential. Sessions are local, in memory, and expire when the server restarts; download a replay to retain a game. The local service retains 128 sessions and limits each to 2,000 placements. Use a single server worker. Public hosting and authentication are not configured yet.

## Development

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
uv run --locked pytest
```

The committed lockfile fixes dependency versions; `uv sync --locked` refuses to silently update it. Python 3.13 is selected by `.python-version`. No pip activation or global dependency installation is needed.

Read [tutorial 00](docs/tutorials/00-foundation.md) for the setup rationale, [the rules](docs/rules.md) for the exact game, and [the API contract](docs/interfaces.md) for integration. This is a simplified turn-based placement game, with no gravity timer, hold or wall kicks. It is not affiliated with Tetris.

## Offline baseline tournament

```bash
uv run --locked stackcraft tournament --seed-start 1000 --episodes 20 --max-pieces 100 --output runs/development.json
```

This runs random and fixed heuristic players on identical seeds, records every decision and replay, and refuses to overwrite an existing result. See [tutorial02](docs/tutorials/02-baselines.md). Development scores are not held-out model results.

## Expert-labelled data

```bash
uv run --locked stackcraft generate-data --output data/study-v1
```

This generates training/validation episodes from fixed disjoint seed pools, labels moves with bounded search, removes duplicate observations and writes a hashed manifest. Final test seeds remain reserved. See [tutorial03](docs/tutorials/03-data.md).

Optional Clef tooling uses `uv sync --locked --extra ml`. Its native encoder, real GPU inference, finite head/LoRA gradients and exact fresh-process checkpoint reload have been verified. Full-study quality is a separate evaluation. See [native adapter documentation](docs/clef.md).

## The model used here

The base is **Cloudflare/clef-flash**, built on Qwen3.5-9B, pinned at
`17f0b0ad64efb65d273590632833508766b2aae6`. It produces probabilities over every
supplied legal move instead of generating text. Training updates rank-4 text-layer
LoRA matrices and the native joint decision head; the original backbone is frozen.
The checkpoint must save and load both the adapter and head.

On the RTX5090, the short native inference pilot allocated19.7GB and averaged144ms
per decision. The training probe allocated23.2GB; both head-only and LoRA-plus-head
checkpoints reproduced their reference probabilities exactly in fresh processes.
These are engineering measurements, not evidence of better game play. See
[the measured feasibility report](reports/gpu-feasibility.json).

## Follow the milestones

1. [Foundation and modern Python tools](docs/tutorials/00-foundation.md)
2. [Deterministic game and replay](docs/tutorials/01-game.md)
3. [Cheap and unchanged-model baselines](docs/tutorials/02-baselines.md)
4. [Search-teacher data and leakage checks](docs/tutorials/03-data.md)
5. [Real GPU training and checkpoint fidelity](docs/tutorials/04-gpu-feasibility.md)
6. [Validation selection and paired game evaluation](docs/tutorials/05-evaluation.md)
7. [Release, public reproduction and fresh-download checks](docs/tutorials/06-release.md)

Each tutorial explains the commands, evidence, design choices and alternatives.
The [evaluation protocol](docs/evaluation-protocol.md) freezes200 test sequences
and a200-piece cap before results. It compares the selected model with the native
base, an unchanged FP32-head ablation, random moves and a fast heuristic.

The training runner intentionally accepts only the exact frozen study dataset.
A newly generated dataset can have different provenance hashes even if its labels
match. For exact reproduction use the released dataset or Tutorial03's source-hash
checked reconstruction. A new dataset or hyperparameter study should be recorded
as a new experiment, not quietly substituted into this study.

## Demo hosting

The [Docker deployment guide](docs/demo-hosting.md) builds a CPU-only image with
no model credentials or Torch. Human gameplay is live; model comparisons replay
recorded decisions and probabilities. Playback speed is separate from measured
inference latency. Local packaging has been verified; public hosting is pending.

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Stackcraft is independent
of Cloudflare, Qwen and Tetris.
