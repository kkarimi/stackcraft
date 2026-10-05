from dataclasses import FrozenInstanceError, replace

import pytest

from stackcraft.engine import legal_actions, new_game, place, step
from stackcraft.pieces import piece_at
from stackcraft.schema import COLORS, HEIGHT, PIECES, WIDTH, GameState


def custom_state(piece: str, rows: list[list[int]]) -> GameState:
    return replace(new_game(42), current=piece, board=tuple(tuple(row) for row in rows))


def blank() -> list[list[int]]:
    return [[0] * WIDTH for _ in range(HEIGHT)]


@pytest.mark.parametrize("piece,count", list(zip(PIECES, (17, 9, 34, 17, 17, 34, 34), strict=True)))
def test_empty_board_legal_actions(piece: str, count: int) -> None:
    state = replace(new_game(42), current=piece)
    actions = legal_actions(state)
    assert len(actions) == len({action.id for action in actions}) == count
    assert [(a.rotation, a.x) for a in actions] == sorted((a.rotation, a.x) for a in actions)
    for action in actions:
        assert action.id == f"r{action.rotation}x{action.x}"
        assert len(set(action.cells)) == 4
        assert all(0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in action.cells)
        assert max(y for _, y in action.cells) == HEIGHT - 1
        after = step(state, action.id).state
        assert sum(cell != 0 for row in after.board for cell in row) == 4
        assert all(after.board[y][x] == COLORS[piece] for x, y in action.cells)
    assert not any(cell for row in state.board for cell in row)


@pytest.mark.parametrize("lines,score", [(1, 100), (2, 300), (3, 500), (4, 800)])
def test_simultaneous_line_clears_and_scoring(lines: int, score: int) -> None:
    rows = blank()
    for y in range(HEIGHT - lines, HEIGHT):
        rows[y] = [7] * WIDTH
        rows[y][4] = 0
    state = custom_state("I", rows)
    result = step(state, "r1x4")
    assert result.cleared == result.state.lines == lines
    assert result.state.score == score
    assert sum(cell != 0 for row in result.state.board for cell in row) == 4 - lines
    assert all(not any(row) for row in result.state.board[:lines])
    assert state.board == tuple(tuple(row) for row in rows)


def test_existing_counters_accumulate_and_rows_keep_order() -> None:
    rows = blank()
    rows[-1] = [7] * 9 + [0]
    rows[-5][0] = 3
    state = replace(custom_state("I", rows), lines=8, score=1600)
    result = step(state, "r1x9")
    assert result.state.lines == 9
    assert result.state.score == 1700
    assert result.state.board[-4][0] == 3


def test_collision_stops_drop_and_does_not_tuck_under_overhang() -> None:
    rows = blank()
    rows[10][0] = 2
    state = custom_state("O", rows)
    action = next(a for a in legal_actions(state) if a.id == "r0x0")
    assert action.y == 8
    assert (0, 10) not in action.cells
    result = step(state, action.id)
    assert result.state.board[10][0] == 2
    assert result.state.board[19][0] == 0


def test_spawn_collision_excludes_a_column_even_if_space_below() -> None:
    rows = blank()
    rows[0][0] = 1
    assert "r0x0" not in {a.id for a in legal_actions(custom_state("O", rows))}


def test_top_out_after_placement_without_completed_rows() -> None:
    rows = blank()
    rows[0] = rows[1] = [0, 0] + [7] * 7 + [0]
    rows[2][0] = 7
    state = replace(custom_state("O", rows), next_piece="O")
    result = step(state, "r0x0")
    assert result.cleared == 0
    assert result.state.terminal
    assert legal_actions(result.state) == ()
    with pytest.raises(ValueError, match="illegal action"):
        step(result.state, "r0x0")


def test_illegal_action_never_mutates_state() -> None:
    state = new_game(10)
    original = new_game(10)
    for action in ("r0x-1", "r0x10", "r99x0", "garbage", "r0x0 "):
        with pytest.raises(ValueError):
            step(state, action)
        assert state == original
    with pytest.raises(FrozenInstanceError):
        state.score = 5  # ty: ignore[invalid-assignment]


def test_piece_advancement_and_repeated_game_equivalence() -> None:
    state = new_game(-53)
    other = new_game(-53)
    for index in range(100):
        assert state.current == piece_at(-53, index)
        assert state.next_piece == piece_at(-53, index + 1)
        actions = legal_actions(state)
        if not actions:
            assert state.terminal
            break
        action = actions[(index * 3) % len(actions)].id
        state = step(state, action).state
        other = step(other, action).state
        assert state == other
        assert state.piece_index == index + 1
        assert len(state.board) == HEIGHT
        assert all(len(row) == WIDTH for row in state.board)
        assert not any(all(row) for row in state.board)


def test_malformed_board_is_rejected() -> None:
    with pytest.raises(ValueError, match="board"):
        replace(new_game(0), board=((0,) * WIDTH,))
    rows = blank()
    rows[0][0] = 8
    with pytest.raises(ValueError, match="cells"):
        custom_state("T", rows)


def test_seedless_placement_matches_step_and_rejects_overlap() -> None:
    state = new_game(42)
    action = legal_actions(state)[0]
    board, cleared = place(state.board, state.current, action)
    transition = step(state, action.id)
    assert board == transition.state.board
    assert cleared == transition.cleared
    with pytest.raises(ValueError, match="empty"):
        place(board, state.current, action)
