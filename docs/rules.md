# Stackcraft v1 rules

Rules version: `stackcraft-v1`. Replay schema version: `1`.

Stackcraft is a turn-based tetromino placement game. Its research rules deliberately
exclude real-time gravity, hold, wall kicks, tucks, lock delay, T-spins, combos,
back-to-back bonuses, and level multipliers. It is not a full competitive Tetris
implementation. Humans and every bot use the same legal-placement enumerator.

## Board and pieces

The board has 10 columns and 20 rows. Coordinates increase rightward (`x`) and
downward (`y`); `(0, 0)` is the top-left cell. Empty cells are `0`. Piece colors
are integer IDs `1..7` corresponding to `I,O,T,S,Z,J,L`.

Each piece has four cells. Orientation zero is defined in `pieces.py`. Other
orientations are successive clockwise rotations, normalized so their minimum x
and y are zero; duplicate orientations are removed. `I,S,Z` have two orientations,
`O` has one, and `T,J,L` have four. Rotation numbers refer to this unique list,
not to a wall-kick rotation system.

## Piece stream and information

Each seven-piece bag contains every shape once. Bag number `b` uses a private
Python `Random` instance seeded with the string `stackcraft-v1:{seed}:{b}` and
an explicitly defined descending Fisher-Yates shuffle: at position `p` swap with
`int(random() * (p + 1))`. This avoids dependence on global random state, request
order, and higher-level shuffle implementation changes. The locked Python
environment and stream fixture tests provide an additional reproduction check.

Players observe the board, current piece, and exactly one next piece. Seeds,
bag state, and later pieces must not appear in model observations. Replays contain
a seed for reconstruction, so replay metadata is not a valid model observation.
Piece index is the count of successfully placed pieces, initially zero.

## Legal moves and top-out

For each unique orientation and fitting column, place the normalized shape at
`y=0`, fully within the board. If any of those cells is occupied, this placement
is unavailable, even if there is space below. Otherwise move the shape down one
row at a time until the next step would collide or leave the board. This is the
only landing for that orientation and column. A piece cannot pass through a
block or slide underneath an overhang.

The action ID is `r{rotation}x{x}`, for example `r1x4`. Legal actions include the
landing y-coordinate and all four absolute cell coordinates. They are ordered
by rotation, then x, making deterministic tie-breaking possible. If there are no
legal moves, the game is terminal. Top-out is evaluated for the next current
piece after the preceding move's row clears. There are no hidden spawn rows.

## Clearing and scoring

After placing a piece, remove every full row simultaneously. Remaining rows keep
their relative order and empty rows are added at the top. Award these points:

| Rows cleared by this move | Points |
| --- | --- |
| 0 | 0 |
| 1 | 100 |
| 2 | 300 |
| 3 | 500 |
| 4 | 800 |

Advance to the preview piece and reveal one new preview. Lines and score are
cumulative; there are no movement or hard-drop points. Lines cleared are the
primary research outcome. Score, placed pieces, and top-out are additional
outcomes. Evaluation episode caps are separate from these rules and must be
reported; reaching a cap does not imply top-out.

## State, errors, and replay

`GameState`, `Placement`, and `Transition` are frozen dataclasses. Boards and
cell collections are tuples. `step(state, action_id)` validates an ID against
the current legal moves and returns a new state. Invalid moves, including any
move after top-out, raise `ValueError`; the input state remains unchanged.

`place(board, piece, action)` is a lower-level helper for expert and heuristic
afterboard evaluation. It reads no seed or future piece. Its action must already
come from `legal_actions` for that board and piece; it checks cell bounds and
occupation, but does not repeat path validation. External move IDs use `step`.

A replay stores schema/rules versions, seed, action IDs, and final score, lines,
placed pieces and terminal status. Import re-simulates every move. Invalid moves,
incompatible versions and inconsistent final summaries are rejected. Partial
games are valid. A consistent replay is reproducible evidence, not a signed
attestation of who played: changing actions and recomputing their summary creates
another valid game.
