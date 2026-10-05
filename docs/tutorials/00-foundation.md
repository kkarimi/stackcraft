# Milestone 00: a reproducible experiment starts with a reproducible game

The eventual question is whether training helps a decision model play better. Before loading a model, establish a game that behaves identically for two players given the same pieces. Otherwise a score difference could reflect easier pieces or different rules.

## 1. Create an isolated Python project

This repository uses Python 3.13, a `src/stackcraft` package, and uv. Run from the repository root:

```bash
uv sync --locked
uv run --locked stackcraft --help
uv run --locked stackcraft --version
```

The help lists `serve`, `tournament` and `generate-data`; the version is `stackcraft 0.1.0`. uv creates `.venv` and installs the package locally. `pyproject.toml` describes dependencies while `uv.lock` records exact resolved versions. Committing both lets another person recreate the environment. `--locked` detects a stale lock instead of silently selecting new versions.

Development dependencies are a dependency group, installed by default, rather than an optional product feature. This keeps pytest and linting separate from game requirements. A future ML extra will isolate Torch and large model dependencies so someone can play the game without a GPU. The code does not import a model from the CLI startup path.

## 2. Write the rules before comparing players

Read `docs/rules.md` and `docs/interfaces.md`. All players receive the same board, current piece, and one-piece preview. They select an orientation and column; Python computes the vertical landing. A fixed seven-bag seed makes piece order reproducible. A seed is stored for replay, but never shown to a model: exposing it or later pieces would let one player gain information unavailable to another.

The simplified rules omit real-time gravity and wall kicks. A full Tetris implementation would mix planning ability with timing and control. We first isolate the decision we want to learn: which legal placement is best? The UI must disclose those rules.

## 3. Keep one authoritative engine

Python simulates placements and emits legal actions with landing cells. The browser draws these cells and sends an action ID back. It does not recalculate collisions. Two engines in different languages could diverge and make replays unreliable. Frozen state objects and pure transition functions make mistakes easier to reproduce and test.

FastAPI serves the small JSON API and static HTML/CSS/JavaScript together. This first version needs no separate Node package manager or frontend framework. That reduces setup friction while leaving room to add one when the interface genuinely needs it.

## 4. Verify behavior and source quality

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
uv run --locked pytest
```

Ruff catches common source mistakes and applies one formatting style. ty checks Python types. pytest verifies outcomes such as a cleared row, an unchanged state after an illegal move, and a replay that reaches the original final board. Passing checks does not prove model quality; that requires a later held-out tournament.

## 5. Preserve decisions and evidence

`plan.md` tracks milestones; `record.md` stores concise results and experiment intentions. Code and tutorials use local Git commits. Environments, weights, raw datasets, credentials and checkpoints are ignored. Later reports will refer to their hashes and release revisions rather than placing gigabytes in Git.

The next milestone adds the engine, playable board and replay. Model training starts only after those behaviors can be verified independently.
