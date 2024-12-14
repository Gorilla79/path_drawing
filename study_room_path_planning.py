import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from scipy.interpolate import splprep, splev
from matplotlib.backend_bases import MouseEvent, KeyEvent

# File path
grid_file_path = r"D:\capstone\24_12_13\study_room_grid_test_size_min.csv"

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
    tck, _ = splprep([x, y], s=1)  # Adjust the smoothing factor 's' if needed
    u_fine = np.linspace(0, 1, len(x) * 10)  # Increase resolution for smooth path
    x_smooth, y_smooth = splev(u_fine, tck)
    return list(zip(x_smooth, y_smooth))

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
        car_size = (3, 6)
        inflated_grid = inflate_obstacles(grid_array, car_size)

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

        # Plot the smoothed path
        fig, ax_map = plt.subplots(figsize=(12, 8))
        ax_map.imshow(grid_array, cmap='gray', origin='upper')
        ax_map.set_title("Smoothed Path")
        ax_map.set_xlabel("X")
        ax_map.set_ylabel("Y")
        smoothed_path_np = np.array(smoothed_path)
        ax_map.plot(smoothed_path_np[:, 0], smoothed_path_np[:, 1], 'b-', linewidth=2, label="Smoothed Path")
        ax_map.legend()
        plt.show()

    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as e:
        print(f"Error loading or processing CSV file: {e}")