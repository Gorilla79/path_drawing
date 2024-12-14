import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.spatial import distance
from heapq import heappush, heappop
from scipy.interpolate import splprep, splev
from matplotlib.backend_bases import MouseEvent, KeyEvent

# File path
grid_file_path = r"D:\capstone\24_12_11\result_grid_test_size_min.csv"

# List to store manually clicked waypoints
clicked_waypoints = []

# Inflate obstacles to account for car size
def inflate_obstacles(grid, car_size):
    """
    Inflate obstacles in the grid to account for the car size.

    Parameters:
        grid (numpy.ndarray): The binary occupancy grid.
        car_size (tuple): (height, width) of the car in grid cells.

    Returns:
        numpy.ndarray: Inflated grid.
    """
    structure = np.ones(car_size)
    inflated_grid = binary_dilation(grid == 0, structure=structure).astype(int)
    return 1 - inflated_grid

# A* Weighted Algorithm with Wall Penalty
def heuristic(a, b):
    return distance.euclidean(a, b)

def astar_weighted(grid, weights, start, goal, distance_map):
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current_cost, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 1:
                wall_penalty = 10 / (1 + distance_map[neighbor]**6)
                new_cost = current_cost + weights[neighbor] + wall_penalty
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

    return []

# Path Smoothing
def smooth_path(path):
    if len(path) < 3:
        return path

    path_np = np.array(path)
    x, y = path_np[:, 1], path_np[:, 0]
    s_factor = max(0.1, len(path) * 0.005)
    tck, _ = splprep([x, y], s=s_factor)
    u_fine = np.linspace(0, 1, len(x) * 100)
    x_smooth, y_smooth = splev(u_fine, tck)
    return list(zip(np.round(y_smooth).astype(int), np.round(x_smooth).astype(int)))

# Check if path is clear
def is_path_clear(path, clearance_grid):
    for point in path:
        if clearance_grid[point] == 0:
            return False
    return True

# Interactive Click Handler
def onclick(event: MouseEvent):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        clicked_waypoints.append((int(y), int(x)))
        waypoint_number = len(clicked_waypoints)
        ax_map.scatter(x, y, color='red', edgecolor='black', s=100)
        ax_map.text(x, y, f"{waypoint_number}", color='blue', fontsize=12, ha='center', va='center')
        print(f"Waypoint {waypoint_number}: (row={int(y)}, column={int(x)})")
        plt.draw()

# Key Press Handler
def on_keypress(event: KeyEvent):
    if event.key == "enter":
        print("Enter pressed. Proceeding with current waypoints...")
        plt.close()

# Distance Calculation
def calculate_path_distance(path, resolution):
    total_distance = 0.0
    for i in range(len(path) - 1):
        current = path[i]
        next_point = path[i + 1]
        pixel_distance = distance.euclidean(current, next_point)
        step_distance = resolution * pixel_distance
        total_distance += step_distance
    return total_distance

# Main Execution
if not os.path.exists(grid_file_path):
    print(f"File not found: {grid_file_path}")
else:
    try:
        grid_data = pd.read_csv(grid_file_path, header=None)
        if grid_data.empty:
            raise ValueError("The CSV file is empty or improperly formatted.")

        grid_data_cleaned = grid_data.fillna(0).applymap(lambda x: 0 if x not in [0, 1] else x)
        grid_array = grid_data_cleaned.to_numpy()

        car_size = (3, 6)
        inflated_grid = inflate_obstacles(grid_array, car_size)

        distance_map = distance_transform_edt(inflated_grid)
        max_distance = distance_map.max()
        weighted_grid = max_distance - distance_map

        print("Click to designate waypoints. Press Enter to proceed after selection.")
        fig, ax_map = plt.subplots(figsize=(12, 8))
        ax_map.imshow(grid_array, cmap='gray', origin='upper')
        ax_map.set_title("Click to Select Waypoints")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        plt.connect('button_press_event', onclick)
        plt.connect('key_press_event', on_keypress)
        plt.show()

        if len(clicked_waypoints) < 2:
            raise ValueError("You must select at least two waypoints.")

        adjusted_paths = []
        path_distances = []
        resolution = 0.1
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'cyan', 'blue', 'darkblue']

        for i in range(len(clicked_waypoints) - 1):
            start = clicked_waypoints[i]
            goal = clicked_waypoints[i + 1]
            path = astar_weighted(inflated_grid, weighted_grid, start, goal, distance_map)
            if path:
                smoothed_path = smooth_path(path)
                if not is_path_clear(smoothed_path, inflated_grid):
                    smoothed_path = path
                adjusted_paths.append(smoothed_path)
                path_distance = calculate_path_distance(smoothed_path, resolution)
                path_distances.append(path_distance)
                print(f"Path {i + 1} Distance: {path_distance:.2f} meters")

        total_distance = sum(path_distances)
        print(f"\nTotal Distance for All Paths: {total_distance:.2f} meters")

        fig, (ax_map, ax_legend) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [10, 1]})
        ax_map.imshow(grid_array, cmap='gray', origin='upper')

        for idx, path in enumerate(adjusted_paths):
            path_x, path_y = zip(*path)
            ax_map.plot(path_y, path_x, color=colors[idx % len(colors)], label=f"Path {idx + 1} ({path_distances[idx]:.2f}m)")

        for wp, coord in enumerate(clicked_waypoints, 1):
            ax_map.scatter(coord[1], coord[0], color='white', edgecolor='black', s=100)
            ax_map.text(coord[1], coord[0], str(wp), color='black', fontsize=10, ha='center', va='center', bbox=dict(boxstyle="circle", fc="white", ec="black", lw=1))

        ax_legend.axis("off")
        ax_legend.legend(handles=ax_map.get_legend_handles_labels()[0], labels=ax_map.get_legend_handles_labels()[1], loc="center", ncol=4, fontsize="small")
        ax_map.set_title("Grid Map with Adjusted Paths")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        plt.show()

    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as e:
        print(f"Error loading or processing CSV file: {e}")