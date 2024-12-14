import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.interpolate import splprep, splev
from heapq import heappush, heappop
from matplotlib.backend_bases import MouseEvent, KeyEvent

# File path
grid_file_path = r"D:\capstone\24_12_13\415insdie_grid_test_size_min.csv"

# List to store drawn path points
drawn_path_points = []

# Inflate obstacles to account for car size
def inflate_obstacles(grid, car_size):
    structure = np.ones(car_size)
    inflated_grid = binary_dilation(grid == 0, structure=structure).astype(int)
    return 1 - inflated_grid

# Smooth the drawn path
def smooth_drawn_path(points):
    if len(points) < 3:
        return points  # No need to smooth if less than 3 points

    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    tck, _ = splprep([x, y], s=5)  # Adjust the smoothing factor 's' if needed
    u_fine = np.linspace(0, 1, len(x) * 10)  # Increase resolution for smooth path
    x_smooth, y_smooth = splev(u_fine, tck)
    return list(zip(x_smooth, y_smooth))

# Find the shortest path between two points while maintaining clearance
def find_shortest_path(start, end, clearance_grid):
    rows, cols = clearance_grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current = heappop(open_set)

        if current == end:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse the path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and clearance_grid[neighbor]:
                new_cost = cost_so_far[current] + 1  # Uniform cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + np.linalg.norm(np.array(neighbor) - np.array(end))
                    heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current

    return []  # No path found

# Adjust the entire drawn path
def adjust_drawn_path(drawn_path, clearance_grid):
    adjusted_path = []
    i = 0
    while i < len(drawn_path):
        start = drawn_path[i]
        # Check for collisions
        if clearance_grid[int(round(start[1])), int(round(start[0]))] == 0:
            # Find the start of the collision
            collision_start = i
            while i < len(drawn_path) and clearance_grid[int(round(drawn_path[i][1])), int(round(drawn_path[i][0]))] == 0:
                i += 1
            # Find the end of the collision
            collision_end = i if i < len(drawn_path) else len(drawn_path) - 1

            # Find a shortest path around the collision
            shortest_path = find_shortest_path(
                (int(round(drawn_path[collision_start][1])), int(round(drawn_path[collision_start][0]))),
                (int(round(drawn_path[collision_end][1])), int(round(drawn_path[collision_end][0]))),
                clearance_grid
            )
            adjusted_path.extend([(p[1], p[0]) for p in shortest_path])  # Append shortest path
        else:
            adjusted_path.append(start)  # Add the point if no collision
            i += 1

    return adjusted_path

# Interactive drawing handler
def on_mouse_drag(event: MouseEvent):
    if event.button == 1 and event.xdata and event.ydata:
        drawn_path_points.append((event.xdata, event.ydata))
        ax_map.plot(event.xdata, event.ydata, 'ro', markersize=2)
        plt.draw()

# Key Press Handler to end drawing
def on_keypress(event: KeyEvent):
    if event.key == "enter":
        print("Enter pressed. Finalizing the path...")
        plt.close()

# Main Execution
if not os.path.exists(grid_file_path):
    print(f"File not found: {grid_file_path}")
else:
    try:
        # Load the CSV file
        grid_data = pd.read_csv(grid_file_path, header=None)
        if grid_data.empty:
            raise ValueError("The CSV file is empty or improperly formatted.")

        # Clean and convert grid data
        grid_data_cleaned = grid_data.fillna(0).applymap(lambda x: 0 if x not in [0, 1] else x)
        grid_array = grid_data_cleaned.to_numpy()

        # Inflate obstacles for car size
        car_size = (6, 6)
        inflated_grid = inflate_obstacles(grid_array, car_size)

        # Compute clearance grid
        clearance_grid = distance_transform_edt(inflated_grid) > 6  # Clearance of 6 cells

        # Interactive Drawing
        print("Click and drag to draw a path. Press Enter to finalize.")
        fig, ax_map = plt.subplots(figsize=(12, 8))
        ax_map.imshow(grid_array, cmap='gray', origin='upper')
        ax_map.set_title("Draw Your Path")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        plt.connect('motion_notify_event', on_mouse_drag)
        plt.connect('key_press_event', on_keypress)
        plt.show()

        # Process the drawn path
        if len(drawn_path_points) < 2:
            raise ValueError("You must draw a valid path with at least two points.")

        # Smooth the path
        smoothed_path = smooth_drawn_path(drawn_path_points)

        # Adjust the smoothed path to maintain clearance
        adjusted_path = adjust_drawn_path(smoothed_path, clearance_grid)

        # Plot the adjusted path
        fig, ax_map = plt.subplots(figsize=(12, 8))
        ax_map.imshow(grid_array, cmap='gray', origin='upper')
        ax_map.set_title("Adjusted Path with Clearance")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        smoothed_path_np = np.array(smoothed_path)
        adjusted_path_np = np.array(adjusted_path)
        ax_map.plot(smoothed_path_np[:, 0], smoothed_path_np[:, 1], 'b-', linewidth=2, label="Smoothed Path")
        ax_map.plot(adjusted_path_np[:, 0], adjusted_path_np[:, 1], 'r-', linewidth=2, label="Adjusted Path")
        ax_map.legend()
        plt.show()

    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as e:
        print(f"Error loading or processing CSV file: {e}")
