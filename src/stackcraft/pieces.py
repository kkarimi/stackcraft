"""Normalized tetromino rotations and the versioned deterministic seven-bag."""

from functools import lru_cache
from random import Random

from stackcraft.schema import PIECES, RULES_VERSION, Cells

_SHAPES: dict[str, Cells] = {
    "I": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "T": ((1, 0), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "Z": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "J": ((0, 0), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 0), (0, 1), (1, 1), (2, 1)),
}


def _normalize(cells: Cells) -> Cells:
    left = min(x for x, _ in cells)
    top = min(y for _, y in cells)
    return tuple(sorted((x - left, y - top) for x, y in cells))


@lru_cache(maxsize=7)
def rotations(piece: str) -> tuple[Cells, ...]:
    """Return distinct clockwise rotations, normalized to top-left (0, 0)."""
    if piece not in _SHAPES:
        raise ValueError(f"unknown piece: {piece!r}")
    current = _normalize(_SHAPES[piece])
    result: list[Cells] = []
    for _ in range(4):
        if current not in result:
            result.append(current)
        current = _normalize(tuple((-y, x) for x, y in current))
    return tuple(result)


@lru_cache(maxsize=4096)
def _bag(seed: int, index: int) -> tuple[str, ...]:
    # Each bag has its own local PRNG: random access does not depend on call order.
    # Freeze the shuffle algorithm as Fisher-Yates using Random.random(), whose
    # compatible-seeder sequence Python guarantees, rather than randrange().
    rng = Random(f"{RULES_VERSION}:{seed}:{index}")
    bag = list(PIECES)
    for position in range(len(bag) - 1, 0, -1):
        other = int(rng.random() * (position + 1))
        bag[position], bag[other] = bag[other], bag[position]
    return tuple(bag)


def piece_at(seed: int, index: int) -> str:
    """Read an indexed piece without revealing or advancing global RNG state."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(index) is not int or index < 0:
        raise ValueError("piece index must be a nonnegative integer")
    bag_index, offset = divmod(index, len(PIECES))
    return _bag(seed, bag_index)[offset]
