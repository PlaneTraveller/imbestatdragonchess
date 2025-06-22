#!/usr/bin/env python3

import numpy as np
from game import Grid
from functools import partial as pt


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


def is_match(board, swap):
    # swap is a tuple of (coord, coord)
    # return true if it's a valid swap
    tmp_grid = board.copy()
    tmp_grid[swap[0]], tmp_grid[swap[1]] = tmp_grid[swap[1]], tmp_grid[swap[0]].copy()

    coord1, coord2 = swap[0], swap[1]

    triplets = get_triplets(coord1) + get_triplets(coord2)

    for triplet in triplets:
        a, b, c = triplet
        if tmp_grid[a] == tmp_grid[b] == tmp_grid[c]:
            return True

    return False


def get_possible_swaps(board):
    rows, cols = board.shape
    possible_swaps = []

    for i in range(rows):
        for j in range(cols):
            # Right neighbor
            if j < cols - 1:
                possible_swaps.append(((i, j), (i, j + 1)))
            # Bottom neighbor
            if i < rows - 1:
                possible_swaps.append(((i, j), (i + 1, j)))

    return possible_swaps


def get_swaps(board):
    # Takes a board, return all possible swap locations (only consider down and right swaps)
    # 7 * 8 * 2 possible swaps, see if any swap result in a match

    possible_swaps = get_possible_swaps(board)
    return list(filter(pt(is_match, board), possible_swaps))


# =======================================================================================
# = Main

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
