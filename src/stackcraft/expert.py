"""Bounded search over the current piece and one visible preview."""

from stackcraft.engine import legal_actions, place
from stackcraft.players import Decision, Observation, board_value
from stackcraft.schema import GameState


class SearchExpert:
    """Two-placement search; its heuristic leaves are not globally optimal labels."""

    name = "search-expert"
    revision = "search-v1-preview1-height1-holes4-bump1-lines8-topout10000"

    def action_values(self, observation: Observation) -> dict[str, int]:
        values = {}
        for action in observation.legal_actions:
            afterboard, cleared = place(observation.board, observation.current, action)
            # Dummy seed/index are never advanced or read. Only preview placements
            # are enumerated; the piece after that preview is deliberately unknown.
            preview_state = GameState(
                board=afterboard,
                seed=0,
                piece_index=0,
                current=observation.next_piece,
                next_piece="I",
            )
            previews = legal_actions(preview_state)
            future = max(
                (
                    board_value(*place(afterboard, observation.next_piece, move))
                    for move in previews
                ),
                default=-10000,
            )
            values[action.id] = 8 * cleared + future
        return values

    def choose(self, observation: Observation) -> Decision:
        values = self.action_values(observation)
        if not values:
            raise ValueError("cannot choose with no legal actions")
        return Decision(max(values, key=lambda action_id: values[action_id]))
