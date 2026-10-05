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

## Watch recorded players side by side

Choose **Watch a race** to load the packaged recordings for unchanged Clef-flash,
the Stackcraft fine-tune and the heuristic. The release's `baseline-demo.json`
contains their actual action traces for **seed 30000**, fixed before inspecting
results, with the same **200-piece cap**. The service reconstructs every frame
through the authoritative replay engine. This replaces the earlier development
Random/Heuristic example; it does not rerun model inference in your browser.

At the final frame of this illustrative seed, unchanged Clef has topped out after
23 pieces without clearing a line, the fine-tune has topped out after 82 pieces
with 16 lines, and the heuristic has reached the 200-piece cap with 78 lines. These
are one game's outcomes, not the aggregate result across 200 held-out seeds.
Consult the full study report to assess overall performance and uncertainty.

The boards are synchronized by placements, not decision time. Playback is
explicitly recorded and does not imply live inference or model speed. A player
that tops out keeps its final board visible while other recordings continue.
Expand a recorded choice to inspect its actual selected-action probability and
up to three alternatives; these are model outputs, not generated explanations.
The provenance section lists the source artifact hashes. Returning to Play
preserves the human game.

## Play alongside recorded opponents

The race view offers an explicit **Start live board · seed N** action. This starts
a new human game with the recording's seed and episode cap; merely opening or
leaving the race tab does not reset the current game. The human board is labelled
**LIVE HUMAN**, while opponents remain labelled as recorded. Both use the Python
engine's legal placements and the same deterministic piece stream.

Each human placement advances the recorded timeline to the same piece count. A
recording that ended earlier stays on its final frame. You can scrub the recorded
timeline independently without changing your board; **Match my move** resynchronizes
it. The original board DOM and session controller move between views, avoiding a
second simulator or conflicting copies of live state.

At the recording's episode cap, human placement controls stop and show **Cap
reached**, distinct from top-out. Switching to Play retains that comparison cap
and the same session. **Leave comparison** removes comparison mode and returns
the existing game to ordinary play; only starting/restarting a game resets it.
Pending moves temporarily disable view changes so a late response cannot update
a different session. Desktop boards align side by side; narrow screens stack them
without shrinking the human controls.
