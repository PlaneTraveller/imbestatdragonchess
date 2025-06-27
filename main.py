import time
import numpy as np
import pyautogui as pa

from game import Grid
from locale_grid import Screen_interact, canny


grid = Grid()
current_screen = np.array(pa.screenshot())
edges = canny(current_screen)
screen = grid.sc.scan_all(current_screen)
grid.sc.calibrate_center(edges, coarse_grid = screen)

while True:
    grid.step()

    # time.sleep(0.1)
