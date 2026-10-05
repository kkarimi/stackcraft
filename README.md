# Stackcraft

A playable, deterministic falling-block game for studying decision models. The goal is a fair comparison of a heuristic player, unchanged Clef-flash, and a fine-tuned model on identical piece sequences.

**Status:** playable game, recorded side-by-side race, offline baselines and search-expert data implemented. The unchanged Clef GPU development baseline now runs; training feasibility is being checked. Final held-out evaluation is pending. See [the project plan](plan.md) for milestones and acceptance criteria.

## Run locally

Install [uv](https://docs.astral.sh/uv/), then from this directory:

```bash
uv sync --locked
uv run --locked stackcraft serve
```

Open http://127.0.0.1:8080. The game needs no GPU or model credential. Sessions are local, in memory, and expire when the server restarts; download a replay to retain a game. The local service retains 128 sessions and limits each to 2,000 placements. Use a single server worker. Public hosting and authentication are not configured yet.

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

Optional Clef tooling uses `uv sync --locked --extra ml`. Its native encoder and real-weight GPU inference have been checked; full-model training probes are in progress. See [native adapter documentation](docs/clef.md).
