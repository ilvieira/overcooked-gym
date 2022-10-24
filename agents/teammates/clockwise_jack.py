from environment.overcooked import ACTION_MEANINGS
from agents.teammates.jack_of_all_trades import JackOfAllTrades


class ClockwiseJack(JackOfAllTrades):
    """Agent that fetches the ingredients, puts the onions in the pan, gets the soup when ready and serves the soup. A
    true Jack of all trades"""

    def __init__(self, layout, index):
        super().__init__(layout, index)


    # only moves the agent in a clockwise manner. The layout is supposed to be the simple kitchen, otherwise, this will
    # not work as expected
    def _action_to_move_to(self, state, target, alternative_target=None, is_teammate_obstacle=True):
        """ Returns the action index that allows the agent to get closer to the target. If the agent is already by the
        target, the action will be to turn arround"""
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        player = (a0_row, a0_column) if self.index == 0 else (a1_row, a1_column)

        if self._cells_are_adjacent(player, target):
            action_meaning = self._turn_to(state, target)
        else:
            action_meaning = self._move_until_adjacent_to(state, target)

        return ACTION_MEANINGS.index(action_meaning)

    def _bottom_row(self, state):
        """returns True iff the agent is in the row 3."""
        return state[0 + 2*self.index] == 3

    def _top_row(self, state):
        """returns True iff the agent is in the row 3."""
        return state[0 + 2*self.index] == 1

    def _left_col(self, state):
        """returns True iff the agent is in the row 3."""
        return state[1 + 2*self.index] == 1

    def _right_col(self, state):
        """returns True iff the agent is in the row 3."""
        return state[1 + 2*self.index] == 4

    @staticmethod
    def _cells_are_adjacent(cell1, cell2):
        return (cell1[0] == cell2[0] and (cell1[1] == cell2[1] + 1 or cell1[1] == cell2[1] - 1)) or \
               (cell1[1] == cell2[1] and (cell1[0] == cell2[0] + 1 or cell1[0] == cell2[0] - 1))

    def _move_until_adjacent_to(self, state, target):
        """to use to move the agent until it is in a cell adjacent to the target"""
        if self._bottom_row(state):
            if self._left_col(state):
                return "up"
            else:
                return "left"
        elif self._top_row(state):
            if self._right_col(state):
                return "down"
            else:
                return "right"
        elif self._right_col(state):
            return "down"
        else:
            return "up"

    def _turn_to(self, state, target):
        """to use when the target is adjacent to the cell the agent wants to reach"""
        a0_row, a0_column, a1_row, a1_column, a0_heading, a1_heading, a0_hand, a1_hand, pan = state[:9]
        player = (a0_row, a0_column) if self.index == 0 else (a1_row, a1_column)

        # same row
        if target[0] == player[0]:
            if player[1] == target[1]+1:
                return "left"
            else:
                return "right"

        # same column
        if target[1] == player[1]:
            if player[0] == target[0]+1:
                return "up"
            else:
                return "down"
