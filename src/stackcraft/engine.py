"""Pure, deterministic vertical-placement falling-block game rules."""

from dataclasses import replace

from stackcraft.pieces import piece_at, rotations
from stackcraft.schema import COLORS, HEIGHT, WIDTH, Board, Cells, GameState, Placement, Transition

_SCORES = (0, 100, 300, 500, 800)


def new_game(seed: int) -> GameState:
    return GameState(
        board=tuple((0,) * WIDTH for _ in range(HEIGHT)),
        seed=seed,
        piece_index=0,
        current=piece_at(seed, 0),
        next_piece=piece_at(seed, 1),
    )


def _fits(state: GameState, cells: Cells, x: int, y: int) -> bool:
    return all(
        0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT and state.board[y + dy][x + dx] == 0
        for dx, dy in cells
    )


def legal_actions(state: GameState) -> tuple[Placement, ...]:
    """Enumerate rotation then column; each move starts fully inside row zero."""
    if state.terminal:
        return ()
    result = []
    for rotation, cells in enumerate(rotations(state.current)):
        width = max(x for x, _ in cells) + 1
        for x in range(WIDTH - width + 1):
            if not _fits(state, cells, x, 0):
                continue
            y = 0
            while _fits(state, cells, x, y + 1):
                y += 1
            absolute = tuple((x + dx, y + dy) for dx, dy in cells)
            result.append(Placement(f"r{rotation}x{x}", rotation, x, y, absolute))
    return tuple(result)


def place(board: Board, piece: str, action: Placement) -> tuple[Board, int]:
    """Place a previously validated action without reading any future pieces.

    Callers must supply an action from legal_actions for this board and piece.
    This low-level helper checks occupied/boundary cells but not hard-drop paths;
    user-controlled IDs must go through step instead.
    """
    if piece not in COLORS:
        raise ValueError(f"unknown piece: {piece!r}")
    if len(set(action.cells)) != 4 or any(
        not (0 <= x < WIDTH and 0 <= y < HEIGHT) or board[y][x] for x, y in action.cells
    ):
        raise ValueError("placement cells must be four distinct empty in-bounds cells")
    rows = [list(row) for row in board]
    for x, y in action.cells:
        rows[y][x] = COLORS[piece]
    remaining = [tuple(row) for row in rows if not all(row)]
    cleared = HEIGHT - len(remaining)
    if cleared > 4:
        raise ValueError("invalid starting board: more than four completed rows")
    return tuple([(0,) * WIDTH] * cleared + remaining), cleared


def step(state: GameState, action_id: str) -> Transition:
    """Validate, place, clear simultaneously, advance, and detect next top-out."""
    action = next((move for move in legal_actions(state) if move.id == action_id), None)
    if action is None:
        raise ValueError(f"illegal action {action_id!r}")
    board, cleared = place(state.board, state.current, action)
    next_state = GameState(
        board=board,
        seed=state.seed,
        piece_index=state.piece_index + 1,
        current=state.next_piece,
        next_piece=piece_at(state.seed, state.piece_index + 2),
        score=state.score + _SCORES[cleared],
        lines=state.lines + cleared,
    )
    if not legal_actions(next_state):
        next_state = replace(next_state, terminal=True)
    return Transition(next_state, action, cleared)
