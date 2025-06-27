import os
import cv2
import glob
import time
import random
import string
import shutil
import numpy as np
import colored
from colored import stylize
from typing import Optional

def canny(image: np.array):
    original_image_for_drawing = image.copy()
    img_h, img_w = image.shape[:2]

    roi = image
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(blurred_roi, 30, 70) # Adjusted thresholds slightly

    return edges

def print_hist(hist_r, hist_g, hist_b):
    import matplotlib.pyplot as plt

    # Plot the histogram
    plt.figure()
    plt.title("Histogram R")
    plt.xlabel("Bins (Pixel Value)")
    plt.ylabel("# of Pixels")
    plt.plot(hists_r)
    plt.xlim([0, 32])
    plt.show()

    plt.figure()
    plt.title("Histogram G")
    plt.xlabel("Bins (Pixel Value)")
    plt.ylabel("# of Pixels")
    plt.plot(hists_g)
    plt.xlim([0, 32])
    plt.show()

    plt.figure()
    plt.title("Histogram B")
    plt.xlabel("Bins (Pixel Value)")
    plt.ylabel("# of Pixels")
    plt.plot(hists_b)
    plt.xlim([0, 32])
    plt.show()

def _ransac_1d(grid_indices, observed_centers, n_iterations, threshold, min_samples):
    """
    Helper function for 1D RANSAC.
    Fits a model: observed_center = c0 + grid_index * dc
    Returns the model parameters (c0, dc) from the best iteration and indices of inliers.
    """
    best_model_params = None
    best_inliers_indices = np.array([], dtype=int)
    max_inliers_count = 0

    if len(grid_indices) < min_samples:
        return None, best_inliers_indices

    data = np.column_stack((grid_indices, observed_centers))

    for _ in range(n_iterations):
        try:
            sample_indices = np.random.choice(len(data), min_samples, replace=False)
        except ValueError: # Not enough points to sample (should be caught by len(grid_indices) < min_samples)
            continue 
        
        samples = data[sample_indices]
        
        s0, s1 = samples[0, 0], samples[1, 0]  # grid_indices from sample
        o0, o1 = samples[0, 1], samples[1, 1]  # observed_centers from sample

        if np.isclose(s0, s1):  # Avoid division by zero; degenerate sample
            continue
        
        # Fit model: c0 + s * dc = o
        dc_candidate = (o0 - o1) / (s0 - s1)
        c0_candidate = o0 - s0 * dc_candidate
        
        # Count inliers
        residuals = np.abs(c0_candidate + data[:, 0] * dc_candidate - data[:, 1])
        inliers_mask = residuals < threshold
        current_inliers_count = np.sum(inliers_mask)
        
        if current_inliers_count > max_inliers_count:
            max_inliers_count = current_inliers_count
            best_model_params = (c0_candidate, dc_candidate)
            best_inliers_indices = np.where(inliers_mask)[0]
    
    return best_model_params, best_inliers_indices

def _least_squares_1d(grid_indices_inliers, observed_centers_inliers):
    """
    Helper function for 1D least squares.
    Fits: observed_center = c0 + grid_index * dc
    Returns (c0, dc)
    """
    if len(grid_indices_inliers) < 2: # Need at least 2 points to define a line
        return None, None
        
    # Design matrix A: [1, grid_index]
    A = np.vstack([np.ones_like(grid_indices_inliers), grid_indices_inliers]).T
    b = observed_centers_inliers
    
    try:
        # Solve A * params = b for params = [c0, dc]^T
        params, residuals_sum_sq, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        if rank < 2 : # Underdetermined (e.g., all grid_indices_inliers are the same)
            return None, None
        c0_fit, dc_fit = params
        return c0_fit, dc_fit
    except np.linalg.LinAlgError:
        return None, None

import pyautogui as pa
import time

base_template_dir = "./templates/"

def get_template_hists(bins = 32, offset = 15):
    # Get all subdirectories (these are your class names)
    # class_folders = [d for d in glob.glob(os.path.join(base_template_dir, '*')) if os.path.isdir(d)]
    # class_folders.sort() # Ensure consistent order
    class_folders = ['./templates/ice_dragon', './templates/lich', './templates/lina', './templates/dawn', './templates/spider', './templates/vs']
    for p in class_folders:
        if not os.path.exists(p):
            os.makedirs(p)
    
    class_names = [os.path.basename(folder) for folder in class_folders]
    hists_r, hists_g, hists_b = [], [], []
    color_directions = []
    
    print(f"Found classes: {class_folders}")
    
    for class_folder in class_folders:
        # Initialize aggregated histograms for the current class
        agg_hist_r = np.zeros((bins, 1), dtype=np.float32)
        agg_hist_g = np.zeros((bins, 1), dtype=np.float32)
        agg_hist_b = np.zeros((bins, 1), dtype=np.float32)
    
        template_files = glob.glob(os.path.join(class_folder, '*.png')) # You can add more extensions like '*.jpg'
        print(f"  Processing {len(template_files)} templates for class: {os.path.basename(class_folder)}")
    
        if not template_files:
            print(f"  Warning: No templates found in {class_folder}")
            # Add empty or some default histogram if a class folder is empty, or handle as error
            # For now, let's add the zero histograms, they will normalize to zero.
            hists_r.append(agg_hist_r)
            hists_g.append(agg_hist_g)
            hists_b.append(agg_hist_b)
            continue
    
        color_direction = np.zeros(3)
        for t_name in template_files:
            template_img = cv2.imread(t_name)
            template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
            template_img = template_img[offset:-offset, offset:-offset, :]
            # aggregate average color in ROI
            color_direction += template_img.mean(axis=(0,1))
            if template_img is None:
                print(f"    Warning: Could not read image {t_name}")
                continue
    
            # OpenCV loads images in BGR order
            r, g, b = cv2.split(template_img)
    
            # Calculate histograms for current image
            hist_r_single = cv2.calcHist([r], [0], None, [bins], [0, 256])
            hist_g_single = cv2.calcHist([g], [0], None, [bins], [0, 256])
            hist_b_single = cv2.calcHist([b], [0], None, [bins], [0, 256])
    
            # Add to the aggregated histogram for the class
            agg_hist_r += hist_r_single
            agg_hist_g += hist_g_single
            agg_hist_b += hist_b_single
    
        hists_r.append(agg_hist_r)
        hists_g.append(agg_hist_g)
        hists_b.append(agg_hist_b)
        color_direction /= len(template_files)
        color_directions.append(color_direction)
    
    for i in range(len(hists_r)):
        cv2.normalize(hists_r[i], hists_r[i], 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(hists_g[i], hists_g[i], 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(hists_b[i], hists_b[i], 0, 255, cv2.NORM_MINMAX)
    
        # print_hist(hists_r[i], hists_g[i], hists_b[i])
    return hists_r, hists_g, hists_b, color_directions

hists_r, hists_g, hists_b, color_directions = get_template_hists(offset=25)

class Screen_interact():
    def __init__(self, y_box_num=8, x_box_num=8, fast_flag = True):
        self.y_box_num = y_box_num
        self.x_box_num = x_box_num

        print("put your mouse at the left top corner of the grid...")
        time.sleep(3)
        uncertain_left_top = pa.position()
        uncertain_left_top = (uncertain_left_top[1], uncertain_left_top[0])
        print(uncertain_left_top)
        print("put your mouse at the right bottom corner of the grid...")
        time.sleep(4)
        uncertain_right_bot = pa.position()
        uncertain_right_bot = (uncertain_right_bot[1], uncertain_right_bot[0])
        print(uncertain_right_bot)
        # uncertain_left_top  = (868, 540)
        # uncertain_right_bot = (1614, 1270)

        self.top_left, self.bot_right = self.locate_grid(uncertain_left_top, uncertain_right_bot)

        self.length_each_y = (self.bot_right[0] - self.top_left[0]) / self.y_box_num
        self.length_each_x = (self.bot_right[1] - self.top_left[1]) / self.x_box_num
        print(f"Each box is {self.length_each_y}px in height and {self.length_each_x}px in width.")

        self.is_fast_flag = fast_flag
        self.flag_memory = [] # used for the fast_flag method.

    def get_border_from_corner(self, edges, left_top, right_bot, check_length=30):
        top_y, left_x = left_top
        bot_y, right_x = right_bot
        res_top_y, res_left_x, res_bot_y, res_right_x = None, None, None, None

        # locate the INNER corner of ┌ and ┘ which have the wanted color
        for y in range(top_y, top_y + check_length):
            if (edges[y, left_x:right_x] == 255).any():
                res_top_y = y
                break
        for x in range(left_x, left_x + check_length):
            if (edges[top_y:bot_y, x] == 255).any():
                res_left_x = x
                break
        for y in range(bot_y, bot_y - check_length, -1):
            if (edges[y, left_x:right_x] == 255).any():
                res_bot_y = y
                break
        for x in range(right_x, right_x - check_length, -1):
            if (edges[top_y:bot_y, x] == 255).any():
                res_right_x = x
                break
        # res_left_x -= 1
        # res_bot_y -= 1
        return (res_top_y, res_left_x), (res_bot_y, res_right_x)

    def locate_grid(self, left_top: Optional[tuple[int, int]] = None,
                    right_bot: Optional[tuple[int, int]] = None, 
                    screen_height = 1440, screen_width = 2560):
        check_length = 50 # check 50 pixels edges

        # current_screen = np.array(pa.screenshot())
        # current_screen = cv2.cvtColor(current_screen, cv2.COLOR_BGR2RGB)
        # edges = canny(current_screen)
        # cv2.imwrite("./test/test_screen_online.png",current_screen)
        # cv2.imwrite("./test/test_screen_online_edge.png",edges)

        current_screen = cv2.imread("./test/test_screen_online.png")
        current_screen = cv2.cvtColor(current_screen, cv2.COLOR_BGR2RGB)
        edges = canny(current_screen)

        top_y, left_x = left_top
        bot_y, right_x = right_bot

        (res_top_y, res_left_x), (res_bot_y, res_right_x) = self.get_border_from_corner(
            edges, left_top, right_bot, check_length=50
        )
        # res_left_x -= 1
        # res_bot_y -= 1
        print(f"grid from {(res_top_y, res_left_x)} to {(res_bot_y, res_right_x)}")
        cv2.rectangle(current_screen, (left_x, top_y), (right_x, bot_y), (0, 0, 255), 2)
        cv2.rectangle(current_screen, (res_left_x, res_top_y), (res_right_x, res_bot_y), (0, 255, 0), 2)
        cv2.imwrite(f"./test/temp.png",current_screen)
        return (res_top_y, res_left_x), (res_bot_y, res_right_x)

    def get_box_center(self, idx_y, idx_x):
        pos_y = self.top_left[0] + self.length_each_x * (idx_y + 0.5)
        pos_x = self.top_left[1] + self.length_each_y * (idx_x + 0.5)
        return (pos_y, pos_x)

    def get_single_box_points(self, idx_y, idx_x): 
        # given the index, return the pixel range contains the wanted box.
        y_pos_start = self.top_left[0] + int(self.length_each_y * (idx_y)) + 0
        y_pos_termi = self.top_left[0] + int(self.length_each_y * (idx_y + 1)) - 0
        x_pos_start = self.top_left[1] + int(self.length_each_x * (idx_x)) + 0
        x_pos_termi = self.top_left[1] + int(self.length_each_x * (idx_x + 1)) - 0
        return y_pos_start, y_pos_termi, x_pos_start, x_pos_termi

    def scan_single_box(self, current_screen, idx_y, idx_x, bins, offset):
        y_pos_start, y_pos_termi, x_pos_start, x_pos_termi = self.get_single_box_points(idx_y, idx_x)
        pixels = current_screen[y_pos_start:y_pos_termi, x_pos_start:x_pos_termi]
        pixels = pixels[offset:-offset, offset:-offset, :]

        # r, g, b = cv2.split(pixels)
        # hist_r = cv2.calcHist([r], [0], None, [bins], [0, 256])
        # hist_g = cv2.calcHist([g], [0], None, [bins], [0, 256])
        # hist_b = cv2.calcHist([b], [0], None, [bins], [0, 256])
        # cv2.normalize(hist_r, hist_r, 0, 255, cv2.NORM_MINMAX)
        # cv2.normalize(hist_g, hist_g, 0, 255, cv2.NORM_MINMAX)
        # cv2.normalize(hist_b, hist_b, 0, 255, cv2.NORM_MINMAX)
        #
        #
        # comps_r, comps_g, comps_b = [], [], []
        # for i in range(len(hists_r)):
        #     comps_r.append(cv2.compareHist(hist_r, hists_r[i], cv2.HISTCMP_CHISQR))
        #     comps_g.append(cv2.compareHist(hist_g, hists_g[i], cv2.HISTCMP_CHISQR))
        #     comps_b.append(cv2.compareHist(hist_b, hists_b[i], cv2.HISTCMP_CHISQR))
        # comps_r, comps_g, comps_b = np.array(comps_r), np.array(comps_g), np.array(comps_b)
        # choose_r, choose_g, choose_b = np.argmin(comps_r), np.argmin(comps_g), np.argmin(comps_b)

        from numpy import dot
        from numpy.linalg import norm
        
        cos_sim = lambda a, b: dot(a, b)/(norm(a)*norm(b))

        color_direction = pixels.mean(axis=(0,1))
        comps_color_dir = np.array([cos_sim(template_dir, color_direction) for template_dir in color_directions])
        choose_color_dir = np.argmax(comps_color_dir)

        # if choose_r == choose_g and choose_r == choose_b and choose_r == choose_color_dir:
        #     current_pixel = choose_r
        # else:
        #     # print(f"WARN: RGB and color doesn't have the same idea on ({idx_x}, {idx_y}): {choose_r}, {choose_g}, {choose_b}, {choose_color_dir}")
        #     # do the majority voting
        #     votes = [choose_r, choose_g, choose_b]
        #     from collections import Counter
        #     majority_vote = Counter(votes).most_common(1)[0][0]
        #     if majority_vote == choose_color_dir:
        #         current_pixel = majority_vote # set as unknown
        #     else:
        #         current_pixel = State["UNKNOWN"].value # set as unknown
        current_pixel = choose_color_dir

        return current_pixel

    def save_template_single_box(self, current_screen, idx_y, idx_x, gt=None):
        y_pos_start, y_pos_termi, x_pos_start, x_pos_termi = self.get_single_box_points(idx_y, idx_x)
        pixels = current_screen[y_pos_start:y_pos_termi, x_pos_start:x_pos_termi]
        if gt is not None:
            idx = gt
        else:
            cv2.imshow("test", pixels)
            cv2.waitKey(1)
            idx = int(input("This belongs to: "))
        
        if idx == 0: save_dir_name = "ice_dragon"
        if idx == 1: save_dir_name = "lich"
        if idx == 2: save_dir_name = "lina"
        if idx == 3: save_dir_name = "dawn"
        if idx == 4: save_dir_name = "spider"
        if idx == 5: save_dir_name = "vs"

        def generate_random_string(length):
            characters = string.ascii_letters
            random_string = ''.join(random.choice(characters) for _ in range(length))
            return random_string

        random_name = generate_random_string(10)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./templates/{save_dir_name}/{random_name}.png", pixels)

    def scan_all(self, current_screen, bins = 32, offset = 15):
        screen = np.zeros((self.y_box_num, self.x_box_num), dtype=int)
        screen[:,:] = 10 #State["UNKNOWN"].value

        for idx_y in range(self.y_box_num):
            for idx_x in range(self.x_box_num):
                # y_pos_start, y_pos_termi, x_pos_start, x_pos_termi = self.get_single_box_points(idx_y, idx_x)
                screen[idx_y, idx_x] = self.scan_single_box(current_screen, idx_y, idx_x, bins=bins, offset=offset)
                # current_screen[y_pos_start:y_pos_termi, x_pos_start:x_pos_termi] = np.array([0, 255, 0])
                # print(idx_y, idx_x, self.G.screen[idx_y, idx_x])

        return screen

    def template_all(self):
        class_folders = ['./templates/ice_dragon', './templates/lich', './templates/lina', './templates/dawn', './templates/spider', './templates/vs']
        for p in class_folders:
            if not os.path.exists(p):
                os.makedirs(p)

        # current_screen = np.array(pa.screenshot())
        current_screen = cv2.imread("./test/test_screen_online.png")
        current_screen = cv2.cvtColor(current_screen, cv2.COLOR_BGR2RGB)
        gt = np.array([
            [2, 2, 3, 2, 1, 4, 2, 2],
            [3, 5, 0, 3, 5, 3, 0, 4],
            [1, 2, 5, 0, 5, 2, 3, 3],
            [2, 3, 5, 5, 0, 5, 0, 0],
            [4, 4, 1, 4, 5, 1, 0, 1],
            [0, 2, 4, 1, 1, 3, 4, 1],
            [4, 3, 0, 5, 5, 2, 4, 0],
            [1, 0, 3, 2, 4, 5, 1, 4]
        ])

        screen = np.zeros((self.y_box_num, self.x_box_num))
        screen[:,:] = 10 #State["UNKNOWN"].value

        for idx_y in range(self.y_box_num):
            for idx_x in range(self.x_box_num):
                self.save_template_single_box(current_screen, idx_y, idx_x, gt=gt[idx_y, idx_x])

    def find_center_single(self, edges, idx_y, idx_x):
        y_pos_start, y_pos_termi, x_pos_start, x_pos_termi = self.get_single_box_points(idx_y, idx_x)
        (res_top_y, res_left_x), (res_bot_y, res_right_x) = self.get_border_from_corner(
            edges, (y_pos_start, x_pos_start), (y_pos_termi, x_pos_termi), check_length=50
        )
        if    res_left_x is None or res_right_x is None \
            or res_top_y is None or   res_bot_y is None:
            return (None, None)
        center_x = (res_left_x + res_right_x) / 2
        center_y = (res_top_y  + res_bot_y)   / 2
        # print(f"idx {idx_y} has center {center_y}")

        return (center_y, center_x)

    def calibrate_center(self, edges, coarse_grid):
        """
        Try to find the center for the (0, 0) grid and delta_x & delta_y between each tile.
        We'll obtain equations in form of
        y_0 + idx_y1 * dy = center_y1
        y_0 + idx_y2 * dy = center_y2
        ...
        x_0 + idx_x1 * dx = center_x1
        x_0 + idx_x2 * dx = center_x2
        ...
        Then we'll use RANSAC to solve y_0, x_0, dy, dx.
        """
        # RANSAC parameters (consider making these configurable, e.g., class attributes or method args)
        ransac_iterations = 100
        ransac_threshold = 3.0  # Max distance for a point to be considered an inlier (in pixels/units of center_x/y)
        min_samples_for_line = 2
        symmetric_tile_ids = [0, 2, 3, 4]
        
        # Find grid indices (gy, gx) of symmetric tiles
        mask = np.isin(coarse_grid, symmetric_tile_ids)
        grid_ys_all_indices, grid_xs_all_indices = np.where(mask)

        if len(grid_ys_all_indices) == 0:
            print("Warning: No symmetric tiles found for calibration.")
            return None, None, None, None 

        # Collect data points for RANSAC:
        # For Y dimension: list of (grid_y_index, observed_center_y)
        # For X dimension: list of (grid_x_index, observed_center_x)
        y_dim_grid_indices, y_dim_observed_centers = [], []
        x_dim_grid_indices, x_dim_observed_centers = [], []

        for gy, gx in zip(grid_ys_all_indices, grid_xs_all_indices):
            # find_center_single is assumed to be a method of this class
            center_y, center_x = self.find_center_single(edges, gy, gx) 
            
            if center_y is not None:
                y_dim_grid_indices.append(gy)
                y_dim_observed_centers.append(center_y)
            if center_x is not None:
                x_dim_grid_indices.append(gx)
                x_dim_observed_centers.append(center_x)
        
        y_dim_grid_indices = np.array(y_dim_grid_indices)
        y_dim_observed_centers = np.array(y_dim_observed_centers)
        x_dim_grid_indices = np.array(x_dim_grid_indices)
        x_dim_observed_centers = np.array(x_dim_observed_centers)

        # Initialize results
        y_0, dy = None, None
        x_0, dx = None, None

        # --- Process Y dimension (y_0, dy) ---
        if len(y_dim_grid_indices) >= min_samples_for_line:
            _, inliers_indices_y = _ransac_1d(
                y_dim_grid_indices, y_dim_observed_centers, 
                n_iterations=ransac_iterations, 
                threshold=ransac_threshold, 
                min_samples=min_samples_for_line
            )

            if inliers_indices_y is not None and len(inliers_indices_y) >= min_samples_for_line:
                inlier_grid_ys = y_dim_grid_indices[inliers_indices_y]
                inlier_center_ys = y_dim_observed_centers[inliers_indices_y]
                y_0, dy = _least_squares_1d(inlier_grid_ys, inlier_center_ys)
                if y_0 is None:
                    print("Warning: Least squares failed for Y dimension after RANSAC.")
            else:
                print("Warning: RANSAC found too few inliers for Y dimension.")
        else:
            print(f"Warning: Not enough data points for Y dimension calibration ({len(y_dim_grid_indices)} points).")

        # --- Process X dimension (x_0, dx) ---
        if len(x_dim_grid_indices) >= min_samples_for_line:
            _, inliers_indices_x = _ransac_1d(
                x_dim_grid_indices, x_dim_observed_centers, 
                n_iterations=ransac_iterations, 
                threshold=ransac_threshold, 
                min_samples=min_samples_for_line
            )

            if inliers_indices_x is not None and len(inliers_indices_x) >= min_samples_for_line:
                inlier_grid_xs = x_dim_grid_indices[inliers_indices_x]
                inlier_center_xs = x_dim_observed_centers[inliers_indices_x]
                x_0, dx = _least_squares_1d(inlier_grid_xs, inlier_center_xs)
                if x_0 is None:
                     print("Warning: Least squares failed for X dimension after RANSAC.")
            else:
                print("Warning: RANSAC found too few inliers for X dimension.")
        else:
            print(f"Warning: Not enough data points for X dimension calibration ({len(x_dim_grid_indices)} points).")
            
        new_top_left  =  (int(y_0 - dy / 2), int(x_0 - dx / 2))                                           
        new_bot_right =  (int(y_0 + dy * (self.y_box_num-1+0.5) ), int(x_0 + (self.x_box_num-1+0.5) * dx))
        print(f"calibration end with (y_0, x_0) = ({y_0:.2f}, {x_0:.2f}), (dy, dx) = ({dy:.2f}, {dx:.2f})")
        print(f"self.top_left  calibrated from {self.top_left} to {new_top_left}")
        print(f"self.bot_right calibrated from {self.bot_right} to {new_bot_right}")
        self.top_left  = new_top_left
        self.bot_right = new_bot_right
        self.length_each_y = dy
        self.length_each_x = dx

    def bootstrap(self):
        global hists_r, hists_g, hists_b, color_directions
        current_screen = cv2.imread("./test/test_screen_online.png")
        current_screen = cv2.cvtColor(current_screen, cv2.COLOR_BGR2RGB)
        edges = canny(current_screen)
        
        self.template_all()
        hists_r, hists_g, hists_b, color_directions = get_template_hists()
        screen = self.scan_all(current_screen)


        self.calibrate_center(edges, coarse_grid = screen)
        breakpoint()

        template_dir = "./templates"
        pattern = os.path.join(template_dir, "*")
        
        # Check if the templates directory itself exists to avoid errors with glob
        if not os.path.isdir(template_dir):
            print(f"Directory '{template_dir}' does not exist. Nothing to do.")
        else:
            # Iterate over all items (files and directories) found by glob
            for item_path in glob.glob(pattern):
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)  # Remove file or symbolic link
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)  # Remove directory and its contents recursively
                except Exception as e:
                    print(f"Error removing {item_path}: {e}")

        self.template_all()
        hists_r, hists_g, hists_b, color_directions = get_template_hists(bins=32, offset=25)
        screen = self.scan_all(current_screen, bins=32, offset=25)

    # Mouse Part
    def move(self, idx_y_start, idx_x_start, idx_y_end, idx_x_end):
        center_from = self.get_box_center(idx_y_start, idx_x_start)
        center_to   = self.get_box_center(idx_y_end, idx_x_end)
        pa.moveTo(center_from[1], center_from[0], 0.2) # (y, x) -> (x, y)
        pa.mouseDown(); 
        pa.moveTo(center_to[1], center_to[0], 0.2)
        pa.mouseUp()

    def step(self):
        current_screen = np.array(pa.screenshot())
        edges = canny(current_screen)
        screen = self.scan_all(current_screen)
        return screen

if __name__ == '__main__':
    sc = Screen_interact()
    current_screen = np.array(pa.screenshot())
    edges = canny(current_screen)
    screen = sc.scan_all(current_screen)
    sc.calibrate_center(edges, coarse_grid = screen)
    while True:
        current_screen = np.array(pa.screenshot())
        edges = canny(current_screen)
        screen = sc.scan_all(current_screen)
        time.sleep(1)
