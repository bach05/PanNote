import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import re
import pickle

import sys
sys.path.append('../auto_calibration_tools')
from ..auto_calibration_tools.scripts.camera_laser_calibration.calibrateCamera2LaserPnP import projectPoint2Image

# Define the folder path
folder_path = 'data_trial'

# Load the laser scan data from scan.csv
scan_df = pd.read_csv(os.path.join(folder_path, 'scan.csv'), sep='\t', header=None)

# Define laser specification
laser_spec = {
    'frame_id': "base_link",
    'angle_min': -3.140000104904175,
    'angle_max': 3.140000104904175,
    'angle_increment': 0.005799999926239252,
    'range_min': 0.44999998807907104,
    'range_max': 25.0
}

# Specify the path of the calibration data
file_path = "../scripts/camera_laser_calibration/laser2camera_map.pkl"

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the dictionary from the pickle file
    data = pickle.load(file)


# Function to read drspaam_data2.csv
def read_drspaam_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip().replace('"', '').split(',') for line in lines]


# Define a function to convert a list of strings into a list of tuples containing 3 floats
def strings_to_tuples(strings):
    tuples = []
    for i in range(0, len(strings), 3):
        x = float(re.sub('[^0-9.-]', '', strings[i]))
        y = float(re.sub('[^0-9.-]', '', strings[i + 1]))
        z = float(re.sub('[^0-9.-]', '', strings[i + 2]))
        tuples.append((x, y, z))
    return tuples

# Function to handle mouse click events
def onclick(event):

    if event.inaxes == ax2:
        x, y = event.xdata, event.ydata
        print(f"Clicked at pixel coordinates: x={x}, y={y}")

# Loop through image files
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        image_path = os.path.join(folder_path, filename)

        # Extract image number from filename
        image_number = int(filename.split('_')[1].split('.')[0])

        # Load the image using PIL
        image = Image.open(image_path)

        # Get the corresponding laser scan data
        scan_data = scan_df.iloc[image_number].tolist()[0]

        # Split the string by commas and convert each element to float
        values = [float(val) for val in scan_data.split(',')]

        img_id = values[0]
        print("Image path: ", image_path)
        print("image ID: ", img_id)

        # Convert the list of values to a NumPy array
        ranges = np.array(values)[1:]

        mask = ranges <= laser_spec['range_max']

        # Calculate the angles for each range measurement
        angles = np.arange(laser_spec['angle_min'], laser_spec['angle_max'], laser_spec['angle_increment'])

        ranges = ranges[mask]
        angles = angles[mask]

        # Convert polar coordinates to Cartesian coordinates
        x = np.multiply(ranges, np.cos(angles))
        y = np.multiply(ranges, np.sin(angles))

        points = np.stack([x, y], axis=1)
        points_polar = np.stack([ranges, angles], axis=1)


        # Get the corresponding dr-spaam data
        drspaam_data_file = os.path.join(folder_path, 'drspaam_data2.csv')
        drspaam_data = read_drspaam_data(drspaam_data_file)[image_number]

        # Use list comprehension to convert the list of strings to tuples
        detections = strings_to_tuples(drspaam_data)
        detections = np.array(detections)

        # Create a 1x2 grid of subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the image in the first subplot
        ax1.imshow(image)
        ax1.set_title(f"Image {image_number}")
        ax1.axis('off')  # Turn off axis labels and ticks

        # Plot the laser scan data in the second subplot
        ax2.scatter(points[:,0], points[:,1], marker="o", color="orange", s=4)
        ax2.scatter(detections[:, 0], detections[:, 1], marker="s", color="red", label="dr-spaam")
        ax2.set_title("Laser Scan Data")
        ax2.axis('equal')
        # Plot horizontal x-axis (red line)
        ax2.grid()

        # Connect the mouse click event to the custom function
        fig.canvas.mpl_connect('button_press_event', onclick)

        # Show the plot
        plt.tight_layout()  # Ensure proper spacing between subplots
        plt.legend()
        plt.show()

        # Now you can work with the image, laser scan data, and dr-spaam data as needed
        # For example, you can print them for verification or perform other tasks

        # Close the image
        image.close()
