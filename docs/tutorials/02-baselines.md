# Tutorial 02 — Build a comparison before training

This milestone establishes offline opponents and a replayable tournament. **Clef has not
been benchmarked yet.** These results cannot establish an advantage from fine-tuning.

## Give every player the same information

`players.observe(state)` creates a frozen `Observation`: board, current piece, exactly
one next piece, legal placements, and the rules version. It omits the episode seed,
piece index, and random generator. A seed would reveal the entire future sequence;
giving it to a search player would invalidate the intended experiment. The tournament
retains the seed separately so another person can replay an episode.

Players implement `choose(observation) -> Decision`. A decision identifies one legal
placement and can provide a complete probability distribution. The runner rejects
illegal moves, missing probabilities, nonfinite probabilities, and incorrect totals.
An exception ends that player's episode as an explicit error. A random fallback would
hide deployment failures and turn the measured player into a different policy.

## Start with two cheap opponents

`RandomPlayer(seed=0)` samples uniformly from legal placements using its own random
generator. Its seed is unrelated to the game's sequence seed. The tournament creates
a fresh instance for each episode, so earlier game length cannot alter later choices.

`HeuristicPlayer` scores every legal afterboard with these fixed, untuned weights:

    value = -aggregate_height - 4*holes - bumpiness + 8*cleared_lines

Aggregate height sums the ten column heights. A hole is an empty cell below a filled
cell in its column. Bumpiness sums differences between adjacent column heights. The
line bonus uses lines cleared by this move. All features describe the board after
simultaneous row removal. Ties follow the engine's rotation-then-column ordering.
These intentionally simple integer weights are a declared baseline, not an optimized
or competitive Tetris claim. The heuristic receives the preview but does not use it;
the later search expert will be able to search one additional placement.

We call the authoritative engine's `place` helper to obtain afterboards. Calling
`step` on the full episode state would expose the hidden future sequence, and a
separate simulator could disagree with the game's actual rules.

## Preregister the development pilot

Before running any tournament, the pilot is fixed as: development seeds **0–19**,
100-piece cap, random RNG seed 0, the weights above, and two players (random and
heuristic). Question: does the runner produce repeatable paired games, and does the
simple board heuristic behave plausibly? This is engineering development, not a
held-out model study. Do not put these seeds into the final test pool or use this
pilot to claim model improvement. Stop after 40 episodes or a runner failure.

Expected artifact: `runs/baselines-development.json`. Record mean/median lines,
score, pieces placed, cap-hit and error rates, and decision latency. Acceptance is
valid replays and no unexplained errors; poor heuristic play is reportable, not a
reason to secretly select nicer seeds. Starting commit is
`53db6bd0e7f42723bc63db7ad56c4ac139a04031`, plus the uncommitted baseline
implementation recorded in Git at integration.

## Reproduce the pilot

From the repository, after `uv sync --locked --group dev`:

```bash
uv run --locked python - <<'PY'
import json
from pathlib import Path
from stackcraft.players import HeuristicPlayer, RandomPlayer
from stackcraft.tournament import run_tournament

result = run_tournament({'random': RandomPlayer, 'heuristic': HeuristicPlayer}, range(20), 100)
output = Path('runs/baselines-development.json')
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['summary'], indent=2))
PY
```

Each episode records visible observations, chosen action IDs, optional probabilities,
the complete action replay, final outcomes, explicit errors, and decision times.
`replay_states(episode['replay'])` reconstructs its final board independently. All
players receive identical piece sequences for each seed, even when one dies earlier.

The cap censors survival: 100 pieces means "survived at least 100", not "would die
at 100". Report cap-hit rate alongside survival. Latency times only `choose`, not
rendering, observation construction, engine transition, or replay validation. It
includes Python overhead and is machine dependent. Game content is deterministic;
latency is deliberately excluded when checking repeatability.

## Development pilot result

The preregistered pilot ran on 2026-10-05. All 40 episode replays matched their
recorded final outcomes; neither player produced an error.

| Player | Mean lines | Median lines | Mean score | Mean pieces | Cap-hit rate | Mean decision time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.10 | 0 | 10 | 26.9 | 0% | 0.0016 ms |
| Heuristic | 35.95 | 36 | 3,900 | 100 | 100% | 0.179 ms |

Every heuristic episode reached the cap. This supports using a longer development
cap to estimate the cost of final evaluation; it does not measure unlimited survival.
The pilot ran in approximately 3.37 seconds including independent replay checks
and writing JSON. Timing is illustrative local CPU execution, with no GPU model.
Artifact SHA-256:
`ed4973c69496cdc0ebb5720e3beaae621406988a6d298233c10ab962e1c499af`.
Regeneration preserves game outcomes but changes timing fields and therefore the
full artifact hash.

## What remains before a model comparison

Integrate the pinned native Clef decision head, record actual model probabilities,
verify that no board or legal-action text is truncated, and persist its errors and
latency with the same runner. Use fresh held-out episode seeds for the final study.
A good baseline is essential: a large neural model can lose to a few lines of board
arithmetic, and the report must make that outcome visible.
