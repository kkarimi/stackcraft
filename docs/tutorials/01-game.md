# Milestone 1: build a game whose decisions can be replayed

The deliverable is a playable turn-based game backed by a deterministic Python
engine. Before interpreting a model's performance, we need confidence that every
player follows identical rules and that saved decisions reconstruct the game.
Read [the complete rules](../rules.md) before extending the simulator.

## 1. Install and start the game

From the repository root, install the committed environment and start the server:

```sh
uv sync --locked
uv run --locked stackcraft serve --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`. Choose a seed and start a game. Left/right select a
column, up changes orientation, and Space or Enter drops the selected placement.
Buttons provide the same controls. There is no gravity timer: take as long as
you need. The preview shows exactly one future piece.

Use restart with the same seed to receive the same piece stream. Download the
replay, load it again, and step through its frames. The imported game is simulated
again by Python; the browser does not trust a claimed score in the file.

Why this design: real-time input would mix decision quality with timing and
keyboard control. Placement decisions are an interpretable first experiment.
The browser displays authoritative legal moves rather than maintaining a second
simulator that could drift away from training rules.

## 2. Inspect the pure engine

```python
from stackcraft.engine import legal_actions, new_game, step

before = new_game(seed=42)
move = legal_actions(before)[0]
transition = step(before, move.id)
assert before.piece_index == 0
assert transition.state.piece_index == 1
print(move.id, move.cells, transition.cleared)
```

Every state is immutable. `step` returns a new state and the resolved placement;
it does not modify `before`. This makes it possible to examine multiple candidate
moves from exactly the same position without accidentally advancing the real game.
Actions are ordered by orientation then column so ties can be resolved reproducibly.

The engine starts each candidate shape at row zero and drops it vertically. It
checks every step rather than picking the lowest empty space: the latter would
incorrectly let pieces pass through overhangs. Row clears happen simultaneously,
then the next piece is checked for top-out.

Why not use a mutable board everywhere: shared mutable state makes search branches,
HTTP sessions, and replay easier to contaminate accidentally. Frozen dataclasses
and tuple boards make those mistakes visible. For game-scale boards, clarity is
more valuable than prematurely optimizing away small copies.

## 3. Separate hidden state from observations

The seed determines independently shuffled seven-piece bags. A bag includes every
shape once; it is not seven independent random draws. `piece_at(seed, index)` is
random-access and uses its own PRNG, so opening another game cannot alter a stream.

Only the current piece and one preview are legitimate player inputs. The engine
and replay know the seed to reconstruct games, but a model or search expert must
not exploit it to look further ahead. The lower-level `place` helper returns an
afterboard and row-clear count without reading any piece stream. Later baselines
and the search expert can therefore evaluate a candidate safely.

## 4. Reconstruct and validate a replay

```python
import json

from stackcraft.replay import make_replay, replay_states

artifact = make_replay(42, [move.id])
frames = replay_states(json.loads(json.dumps(artifact)))
assert frames == [before, transition.state]
```

The replay contains actions rather than a pile of unverified board screenshots.
Its rules version is necessary: changing rotations or scoring must not silently
reinterpret an old experiment. Import also checks the final summary against the
recomputed outcome and rejects moves after terminal state.

This detects inconsistent files, not malicious authorship. Someone can construct
a different valid action sequence and summary; publication provenance and artifact
hashes will be separate experiment records.

## 5. Check the behavioural contract

```sh
uv run --locked pytest tests/test_pieces.py tests/test_engine.py tests/test_replay.py
```

These tests cover shape orientations, all empty-board placements, boundaries,
collisions, blocked entry paths, one through four simultaneous row clears,
score accumulation, top-out, deterministic piece streams, state immutability,
replay round-trips and tampering. Tests for the service cover request handling
separately. Browser inspection must additionally check readable ghost pieces,
keyboard/button actions, restart and replay controls; engine tests cannot prove
that an interface is usable.

Next, measure random and heuristic players before adding a model. A playable
game establishes the experimental environment; it does not yet demonstrate that
fine-tuning helps.
