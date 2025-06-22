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
    The grid class for the dragon chess.
    """
    def __init__(self, height=8, width=8):
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)
        self.mouse = None
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




if __name__ == "__main__":
    # grid = Grid()
    # grid.init_randomly()
    # grid._show()

    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    
    img_rgb = cv2.imread("./test/test_screen.png")
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    templates_filename = [
        "./templates/ice_dragon.png",
        "./templates/lich.png",
        "./templates/lina.png",
        "./templates/dawn.png",
        "./templates/spider.png",
        "./templates/vs.png",
    ]
    templates = [ cv2.imread(t_name, cv2.IMREAD_GRAYSCALE) for t_name in templates_filename ]
    
    for i in range(len(templates)):
        img_temp = np.copy(img_rgb)
        w, h = templates[i].shape[::-1]
        res = cv2.matchTemplate(img_gray,templates[i],cv2.TM_CCOEFF_NORMED)
        print(res.min(), res.max())
        # breakpoint()
        threshold = 0.5
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_temp, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        cv2.imwrite(f"./test/res_{i}.png",img_temp)
