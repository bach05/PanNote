import os
import pandas as pd
from PIL import Image

import matplotlib as mpl
import sys

from src.auto_labeling_tools.util.image_detector import ImageDetector

mpl.use('TkAgg')

import matplotlib.pyplot as plt

import numpy as np
import re
import pickle
import cv2

from src.auto_calibration_tools.scripts.camera_laser_calibration.calibrateCamera2LaserPnP import projectPoint2Image
import detect_people
from matplotlib.backend_bases import MouseButton

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
class ClickHandler:
    def __init__(self, read_ax, write_ax, H):
        self.read_ax = read_ax
        self.write_ax = write_ax
        self.H = H
        self.u_left = None
        self.u_right = None
        self.points = None
        self.crop = [0, 0, 0, 0]
        self.crop1 = [0, 0]
        self.crop2 = [0, 0]
        self.line_left = None
        self.line_right = None
        self.centroid_line = None

        self.centroid = [0, 0]
        self.bbox_id = 0
        self.projected_centroid = None
        self.buffer = ""
        self.bbox = []
        self.selected_points = None

        self.result = []

    def set_pointcloud(self, points):
        self.points = points

    def set_bbox(self, boxes):
        self.bbox = boxes

    def get_results(self):
        return self.result

    def onclick(self, event):
        if event.inaxes == self.read_ax:
            if event.dblclick:
                #leggo click left
                if event.button is MouseButton.LEFT:
                    x, y = event.xdata, event.ydata
                    print(f"Clicked at pixel coordinates: x={x}, y={y}")

                    if self.line_left is not None:
                        self.line_left.remove()
                        self.line_left = None
                    if self.centroid_line is not None:
                        self.centroid_line.remove()
                        self.centroid_line = None


                    laser_point = np.array([[x,y]])
                    self.u_left = projectPoint2Image(laser_point, self.H)
                    print(f"Predicted u left {self.u_left}")

                    self.crop[0] = x
                    self.crop[1] = y
                    self.crop1 = [x, y]

                    self.line_left = self.write_ax.axvline(self.u_left, c="blue")
                    if self.selected_points is not None:
                        self.read_ax.scatter(self.selected_points[:, 0], self.selected_points[:, 1], marker="o", color="orange", s=4)


                #leggo click right
                elif event.button is MouseButton.RIGHT:

                    #devo aver settato già il primo corner
                    if self.u_left is None:
                        print("Define the upper left corner before with a LEFT click!")

                    else:

                        x, y = event.xdata, event.ydata
                        print(f"Clicked at pixel coordinates: x={x}, y={y}")

                        if self.line_right is not None:
                            self.line_right.remove()
                            self.line_right = None
                            if self.selected_points is not None:
                                self.read_ax.scatter(self.selected_points[:, 0], self.selected_points[:, 1], marker="o", color="orange", s=4)

                            if self.centroid_line is not None:
                                self.centroid_line.remove()
                                self.centroid_line = None

                        laser_point = np.array([[x, y]])
                        self.u_right = projectPoint2Image(laser_point, self.H)
                        print(f"Predicted u right {self.u_right}")

                        self.crop[2] = x
                        self.crop[3] = y
                        self.crop2 = [x, y]

                        #processing del crop nella pointcloud....
                        self.line_right = self.write_ax.axvline(self.u_right, c="red")

                        self.selected_points = self.points[np.where((self.points[:, 0] > self.crop[0]) & (self.points[:, 0] < self.crop[2]) &(self.points[:,1] > self.crop[3]) & (self.points[:, 1] < self.crop[1]))]
                        self.read_ax.scatter(self.selected_points[:, 0], self.selected_points[:, 1], marker="o", color="blue", s=4)

                        self.centroid = np.mean(self.selected_points, axis=0)
                        self.read_ax.scatter(self.centroid [0], self.centroid [1], marker="x", color="green", s=5)

                        self.rojected_centroid = projectPoint2Image(np.array([self.centroid]), self.H)
                        self.centroid_line = self.write_ax.axvline(self.rojected_centroid, c="green")

                        #reset
                        # self.u_left = None
                        # self.u_right = None


            plt.show()

    def on_press(self, event):
        #print('press', event.key)
        sys.stdout.flush()
        if event.key == 'q':
            print("completed")
            plt.close()
        if event.key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            self.buffer += event.key
            print(self.buffer)
        if event.key == "enter":
            print("assosiation_done")
            image_label = int(self.buffer)
            print("image label")
            print(image_label)
            print("boxes")
            print(self.bbox[image_label])
            print("position")
            print(self.centroid)

            self.result.append([self.bbox[image_label], self.centroid])

            self.read_ax.scatter(self.selected_points[:, 0], self.selected_points[:, 1], marker="o", color="black", s=4)
            self.read_ax.scatter(self.centroid[0], self.centroid[1], marker="x", color="black", s=5)
            self.buffer = ""
            self.line_right.remove()
            self.line_left.remove()
            self.line_right = None
            self.line_left = None

        if event.key == "delete":
            print("reset")
            self.buffer = ""




def main():

    # Define the folder path
    folder_path = "/home/leonardo/Downloads/hospital1_static/"

    # Load the laser scan data from scan.csv
    scan_df = pd.read_csv(os.path.join(folder_path, 'laser.csv'), sep='\t', header=None)

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
    file_path = "../auto_calibration_tools/scripts/camera_laser_calibration/laser2camera_map.pkl"

    # ouput path
    out_file = os.path.join(folder_path, "annotations.csv")
    if os.path.exists(out_file):
        file_annotation = open(out_file, "a")
    else:
        file_annotation = open(out_file, "w")

    # set yolo
    #yolo_model = detect_people.load_model(weights="models/yolov7.pt")
    imde = ImageDetector("models/yolov7.pt")

    mpl.use('TkAgg')


    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the dictionary from the pickle file
        calibration_map = pickle.load(file)

    H = calibration_map['H2']

    # Loop through image files
    image_folder_path = os.path.join(folder_path, "img")

    for index, row in scan_df.iterrows():

        if index < 42:
            continue

        # Split the input string by comma and remove leading/trailing whitespace
        row = row[0].split(',')

        # Optionally, you can remove leading/trailing whitespace from each value in the list
        row = [value.strip('"') for value in row]

        image_index = row[0]
        image_path = os.path.join(image_folder_path, image_index+".png")
        # Load the image using PIL
        image = Image.open(image_path)

        laser_scan = np.array([float(value) for value in row[1:]])

        # Convert the list of values to a NumPy array
        ranges = np.array(laser_scan)

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
        # drspaam_data_file = os.path.join(folder_path, 'drspaam_data2.csv')
        # drspaam_data = read_drspaam_data(drspaam_data_file)[image_number]
        #
        # # Use list comprehension to convert the list of strings to tuples
        # detections = strings_to_tuples(drspaam_data)
        # detections = np.array(detections)

        cv_image = np.array(image)
        detected, _, _ = imde.compute_detections_sides(cv_image)
        #detected, objects_pose = detect_people.detect_person(cv_image, yolo_model)

        for j in range(len(detected)):
            point1 = detected[j][:2]
            point2 = detected[j][2:]
            cv2.rectangle(cv_image, point1, point2, (0, 255, 0), 3)
            cv2.putText(cv_image, str(j), point1, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

        # Create a 1x2 grid of subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        ch = ClickHandler(read_ax=ax2, write_ax=ax1, H=H)
        ch.set_pointcloud(points)
        ch.set_bbox(detected)

        # Plot the image in the first subplot
        ax1.imshow(cv_image)
        ax1.set_title(f"Image {image_index}")
        ax1.axis('off')  # Turn off axis labels and ticks

        # Plot the laser scan data in the second subplot
        ax2.scatter(points[:, 0], points[:, 1], marker="o", color="orange", s=4)
        ax2.set_title("Laser Scan Data")
        ax2.axis('equal')
        # Plot horizontal x-axis (red line)
        ax2.grid()

        # Connect the mouse click event to the custom function
        fig.canvas.mpl_connect('button_press_event', ch.onclick)
        fig.canvas.mpl_connect('key_press_event', ch.on_press)

        # Show the plot
        plt.tight_layout()  # Ensure proper spacing between subplots
        #plt.legend()
        plt.show()

        # Now you can work with the image, laser scan data, and dr-spaam data as needed
        # For example, you can print them for verification or perform other tasks

        # Close the image
        image.close()

        results = ch.get_results()
        print(results)
        file_annotation.write(image_index)
        for e in results:
            box, centroid = e
            file_annotation.write(";"+str(box[0])+";"+str(box[1])+";"+str(box[2])+";"+str(box[3])+";"+str(centroid[0])+";"+str(centroid[1]))
        file_annotation.write("\n")
        file_annotation.flush()
        os.fsync(file_annotation.fileno())





if __name__ == '__main__':
    main()