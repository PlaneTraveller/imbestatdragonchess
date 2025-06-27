import random
import numpy as np
import colored
from colored import stylize
from typing import Optional
from functools import partial as pt
from locale_grid import Screen_interact

from enum import Enum
class State(Enum):
    ICE_DRAGON = 0
    LICH = 1
    LINA = 2
    DAWN = 3
    SPIDER = 4
    VS = 5
    UNKNOWN=10


def get_triplets(coord, rows=8, cols=8):
    x, y = coord[0], coord[1]
    triplets = []

    # Horizontal triplets (left/right)
    if y - 2 >= 0:  # Left triplet: [ (x, y-2), (x, y-1), (x, y) ]
        triplets.append([(x, y - 2), (x, y - 1), (x, y)])
    if y - 1 >= 0 and y + 1 < cols:  # Middle horizontal: [ (x, y-1), (x, y), (x, y+1) ]
        triplets.append([(x, y - 1), (x, y), (x, y + 1)])
    if y + 2 < cols:  # Right triplet: [ (x, y), (x, y+1), (x, y+2) ]
        triplets.append([(x, y), (x, y + 1), (x, y + 2)])

    # Vertical triplets (up/down)
    if x - 2 >= 0:  # Above triplet: [ (x-2, y), (x-1, y), (x, y) ]
        triplets.append([(x - 2, y), (x - 1, y), (x, y)])
    if x - 1 >= 0 and x + 1 < rows:  # Middle vertical: [ (x-1, y), (x, y), (x+1, y) ]
        triplets.append([(x - 1, y), (x, y), (x + 1, y)])
    if x + 2 < rows:  # Below triplet: [ (x, y), (x+1, y), (x+2, y) ]
        triplets.append([(x, y), (x + 1, y), (x + 2, y)])

    return triplets

class Grid:
    """
    The grid class for the dragon chess.
    """
    def __init__(self, height=8, width=8):
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)
        self.sc = Screen_interact()
    def _show(self, show_line_number = True):
        # show the grid of current gamestate.
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
        c = [12, 18, 130, 3, 52, 93] # the color number for colored.
        return_character = "■"
        if state_enum == State["UNKNOWN"].value:
            return_character = "U"
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

    def is_match(self, swap):
        # swap is a tuple of (coord, coord)
        # return true if it's a valid swap
        tmp_grid = self.grid.copy()
        tmp_grid[swap[0]], tmp_grid[swap[1]] = tmp_grid[swap[1]], tmp_grid[swap[0]].copy()
    
        coord1, coord2 = swap[0], swap[1]
        triplets = get_triplets(coord1) + get_triplets(coord2)
    
        for triplet in triplets:
            a, b, c = triplet
            if tmp_grid[a] == tmp_grid[b] == tmp_grid[c]:
                return True

        return False

    def get_possible_swaps(self):
        possible_swaps = []
    
        for i in range(self.height):
            for j in range(self.width):
                # Right neighbor
                if j < self.width - 1:
                    possible_swaps.append(((i, j), (i, j + 1)))
                # Bottom neighbor
                if i < self.height - 1:
                    possible_swaps.append(((i, j), (i + 1, j)))

        return possible_swaps

    def get_swaps(self):
        # Return all possible swap locations for self.grid (only consider down and right swaps)
        # 7 * 8 * 2 possible swaps, see if any swap result in a match
        possible_swaps = self.get_possible_swaps()
        return list(filter(pt(self.is_match), possible_swaps))

    def step(self):
        screen = self.sc.step()
        self.grid = screen
        self._show()
        possible_swaps = self.get_swaps()
        possible_swaps.reverse()
        s = [possible_swaps[0]]
        for i in possible_swaps:
            if abs(i[0][1] - s[0][0][1]) > 4:
                s.append(i)
        for m in s[:2]:
            self.sc.move(m[0][0], m[0][1], m[1][0], m[1][1])
        # s = possible_swaps[-1]
        # self.sc.move(s[0][0], s[0][1], s[1][0], s[1][1])

if __name__ == "__main__":
    grid = Grid()
    grid.init_randomly()
    grid._show()
    # grid.grid[(1, 1)] = 1
    # grid.grid[(2, 1)] = 1
    # grid.grid[(3, 1)] = 1
    # grid.grid[(3, 2)] = 1
    print(grid.grid)

    print(list(get_swaps(grid.grid)))
