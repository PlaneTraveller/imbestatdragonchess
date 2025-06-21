import numpy as np
import colored
from colored import stylize
from typing import Optional


from enum import Enum
class State(Enum):
    ICE_DRAGON = 0
    LICH = 1
    LINA = 2
    DAWN = 3
    SPIDER = 4
    VS = 5
    UNKNOWN=10

class Grid:
    """
    The grid class for the minesweeper.
    """
    def __init__(self, height=8, width=8):
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)
        # print(self.grid)
        self.screen = np.zeros((height, width), dtype=int)
        self.screen[:,:] = State["UNKNOWN"].value
        self.mouse = None
    def _show(self, show_line_number = True):
        # show the screen of current gamestate.
        return_str = "\n"
        if (show_line_number):
            return_str += "| |"
            for i in range(self.width):
                return_str += str(i)[-1] + "|"
            return_str += "\n"
        for i in range(self.height):
            cur_line = "|"
            if (show_line_number):
                cur_line += str(i)[-1] + "|"
            for j in range(self.width):
                cur_word = self.convert_to_display(self.grid[i][j])
                cur_line += (cur_word + "|")
            cur_line += "\n"
            return_str += cur_line
            # print(f"{fg(22)}logger: {attr(0)}{fg(42)}{name}{attr(0)}\t{fg(44)}{value}{attr(0)}")
        print(return_str)

    def convert_to_display(self, state_enum):
        # light_blue, dark_blue, orange, yellow, dark_red, purple
        c = [12, 18, 130, 3, 52, 54] # the color number for colored.
        return_character = "■" #return_character = "●"
        if state_enum == State["UNKNOWN"]:
            return_character = "U" #return_character = "●"
        else: # normal tiles
            return_character = stylize(return_character, colored.fg(f"{c[state_enum]}"))
        return return_character

    def init_randomly(self):
        ### randomly set the tiles ###
        import random
        from datetime import datetime
        random.seed(datetime.now().timestamp())
        
        # total 6 kinds of tiles
        self.grid = np.random.randint(0, 5 + 1, size=(self.height, self.width))

if __name__ == "__main__":
    grid = Grid()
    grid.init_randomly()
    grid._show()
