"""Immutable types shared by the engine, players, replay, and HTTP service."""

from dataclasses import dataclass

WIDTH = 10
HEIGHT = 20
RULES_VERSION = "stackcraft-v1"
PIECES = ("I", "O", "T", "S", "Z", "J", "L")
COLORS = {piece: index + 1 for index, piece in enumerate(PIECES)}
type Cells = tuple[tuple[int, int], ...]
type Board = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Placement:
    id: str
    rotation: int
    x: int
    y: int
    cells: Cells


@dataclass(frozen=True)
class GameState:
    board: Board
    seed: int
    piece_index: int
    current: str
    next_piece: str
    score: int = 0
    lines: int = 0
    terminal: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.board, tuple) or len(self.board) != HEIGHT:
            raise ValueError(f"board must contain {HEIGHT} immutable rows")
        if any(
            not isinstance(row, tuple)
            or len(row) != WIDTH
            or any(type(cell) is not int or not 0 <= cell <= 7 for cell in row)
            for row in self.board
        ):
            raise ValueError(f"board rows must contain {WIDTH} integer cells from 0 to 7")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("piece_index", "score", "lines"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.current not in PIECES or self.next_piece not in PIECES:
            raise ValueError("current and next_piece must be tetromino names")
        if type(self.terminal) is not bool:
            raise ValueError("terminal must be a boolean")


@dataclass(frozen=True)
class Transition:
    state: GameState
    action: Placement
    cleared: int
