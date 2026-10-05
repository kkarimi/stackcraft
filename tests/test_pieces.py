import random

import pytest

from stackcraft.pieces import piece_at, rotations
from stackcraft.schema import PIECES


@pytest.mark.parametrize("piece,count", list(zip(PIECES, (2, 1, 4, 2, 2, 4, 4), strict=True)))
def test_distinct_normalized_connected_rotations(piece: str, count: int) -> None:
    variants = rotations(piece)
    assert len(variants) == count
    assert len(set(variants)) == count
    for cells in variants:
        assert len(set(cells)) == 4
        assert min(x for x, _ in cells) == min(y for _, y in cells) == 0
        reached = {cells[0]}
        for _ in range(3):
            reached |= {
                cell
                for cell in cells
                if any(abs(cell[0] - x) + abs(cell[1] - y) == 1 for x, y in reached)
            }
        assert reached == set(cells)


def test_seven_bags_are_deterministic_random_access_and_local() -> None:
    global_state = random.getstate()
    stream = [piece_at(42, index) for index in range(700)]
    assert [piece_at(42, index) for index in reversed(range(700))] == stream[::-1]
    assert random.getstate() == global_state
    assert all(set(stream[start : start + 7]) == set(PIECES) for start in range(0, 700, 7))
    assert stream != [piece_at(43, index) for index in range(700)]


def test_version_one_stream_fixture() -> None:
    assert "".join(piece_at(42, index) for index in range(28)) == "TIZJSOLTJLSOIZLJOSIZTILOZSJT"


@pytest.mark.parametrize("seed,index", [(True, 0), (1.2, 0), (1, -1), (1, True), (1, 0.1)])
def test_bad_piece_stream_inputs(seed: object, index: object) -> None:
    with pytest.raises(ValueError):
        piece_at(seed, index)  # ty: ignore[invalid-argument-type]


def test_unknown_piece() -> None:
    with pytest.raises(ValueError):
        rotations("Q")
