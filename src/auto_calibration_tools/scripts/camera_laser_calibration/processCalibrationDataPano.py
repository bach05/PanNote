#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from tqdm import tqdm
import cv2

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms.functional as F
from cube_projection_LB import CubeProjection
from PIL import Image, ImageDraw
import pickle
import csv

from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from tqdm import tqdm

import matplotlib as mpl
mpl.use('TkAgg')


def plot_points_on_image(image, points, point_color=(255, 0, 0), output_path=None):
    """
    Plot points on a PIL image.

    Args:
        image_path (str): The path to the input image.
        points (np.ndarray): Numpy array containing points in shape (N, 1, 2).
        point_color (tuple): The color of the points (R, G, B). Default is red.
        output_path (str): The path to save the modified image. If None, the image is not saved.

    Returns:
        PIL.Image.Image: The PIL image with the points plotted.
    """
    # Load the image
    image = Image.fromarray(image)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Draw points on the image
    for point in points:
        x, y = point
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=point_color)

    # Save the modified image if an output path is provided
    if output_path:
        image.save(output_path)

    return image

def draw_division_lines(image, bounding_boxes, scale=1):

    original_image = Image.fromarray(image)
    draw = ImageDraw.Draw(original_image)

    divisions = [
        (0, "back", 0, scale * 720, original_image.width, scale * 720),
        (1, "left", scale * 720, 0, scale * 720, original_image.height),
        (2, "front", scale * 1200, 0, scale * 1200, original_image.height),
        (3, "right", scale * 1680, 0, scale * 1680, original_image.height)
    ]

    for bounding_box in bounding_boxes:
        u1, v1, u2, v2 = bounding_box

        for division in divisions:
            division_id, label, x1, y1, x2, y2 = division
            draw.line([(x1, y1), (x2, y2)], fill="red", width=2)  # You can adjust the line color and width
            print(f"Drawing {label} division line for bounding box {bounding_box}")

        # Draw the bounding box
        draw.rectangle([(u1, v1), (u2, v2)], outline="yellow", width=2)  # Adjust bounding box parameters
        print(f"Drawing bounding box {bounding_box}")

    original_image.show()
def draw_bounding_box(image, bbox, box_color=(255, 0, 0), outline_width=2, output_path=None):
    """
    Draws a bounding box on an image and optionally saves it to a file.

    Args:
        image_path (str): The path to the input image.
        bbox (tuple): The bounding box coordinates (left, top, right, bottom).
        box_color (tuple): The color of the bounding box (R, G, B). Default is red.
        outline_width (int): The width of the bounding box outline. Default is 2.
        output_path (str): The path to save the modified image. If None, the image is not saved.

    Returns:
        PIL.Image.Image: The PIL image with the bounding box drawn.
    """

    # Load the image
    image = Image.fromarray(image)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Draw the bounding box
    draw.rectangle(bbox, outline=box_color, width=outline_width)

    # Save the modified image if an output path is provided
    if output_path:
        image.save(output_path)

    return image


class Reporter:

    def __init__(self):

        self.report = {
            'failures':{
                'chessboard_not_detected': 0,
                'chessboard_not_detected_cropping': 0,
                'corners_not_found': 0,
                'unable_to_cube_project': 0,
                'not_enough_laser_patterns': 0,
                'low_confidence_laser_pattern': 0,
            },
        }

    def updateFailure(self, failure_name):

        self.report['failures'][failure_name] +=1

    def printReport(self, category):

        for key, value in self.report[category].items():
            print("{} : {}".format(key, value))


class Plotter:
    def __init__(self, subplot_shape=(2, 2)):
        self.subplot_shape = subplot_shape
        self.figure, self.axes = plt.subplots(subplot_shape[0], subplot_shape[1], figsize=(16, 12))
        self.current_subplot_index = 0

    def getSubplot(self, index):
        if index < 1 or index > self.subplot_shape[0] * self.subplot_shape[1]:
            raise ValueError("Invalid subplot index")

        row = (index - 1) // self.subplot_shape[1]
        col = (index - 1) % self.subplot_shape[1]

        if self.subplot_shape[0] == 1:
            ax = self.axes[col]
        elif self.subplot_shape[1] == 1:
            ax = self.axes[row]
        else:
            ax = self.axes[row, col]

        self.current_subplot_index = index

        self.current_ax = ax

        return ax

    def zoom_around_point(self, center_x, center_y, radius):
        """
        Zoom the current subplot around a specific point within a given radius.

        Args:
            center_x (float): X-coordinate of the center point.
            center_y (float): Y-coordinate of the center point.
            radius (float): The radius around the center point to include in the zoom.
        """

        x_min, x_max = center_x - radius, center_x + radius
        y_min, y_max = center_y - radius, center_y + radius

        self.current_ax.set_xlim(x_min, x_max)
        self.current_ax.set_ylim(y_min, y_max)


    def show(self):

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()

    def save(self, side, img_id, subfolder="log_imgs"):

        path = os.path.join(side, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(os.path.join(path, img_id+".png"))

        for ax in self.axes:
            ax.clear()

    def clear(self):

        for ax in self.axes:
            ax.clear()


def cartesian_to_polar(cartesian_points):

    x, y = cartesian_points[:, 0], cartesian_points[:, 1]
    angles = np.arctan2(y, x)  # Calculate angles in radians
    angles_deg = np.degrees(angles)  # Convert angles to degrees

    # Map angles to the [0°, 360°] range
    #angles_deg = (angles_deg + 360) % 360
    offset = -50
    for i in range(len(angles_deg)):
        if angles_deg[i] < 0:
            angles_deg[i] = 360 + angles_deg[i]
            offset = 50

    # Calculate magnitudes (distances)
    magnitudes = np.sqrt(x**2 + y**2)

    return np.column_stack((angles_deg, magnitudes))


def map_bounding_box_to_side(bounding_box, scale=1):
    u1, v1, u2, v2 = bounding_box

    if (u1 >= 0 and u2 < scale*240) or (u1 >= scale*1680 and u2 < scale*1920):
        if v1 >= scale*240 and v2 < scale*720:
            return (0, "back")
    elif u1 >= scale*240 and u2 < scale*720:
        if v1 >= scale*240 and v2 < scale*720:
            return (1, "left")
    elif u1 >= scale*720 and u2 < scale*1200:
        if v1 >= scale*240 and v2 < scale*720:
            return (2, "front")
    elif u1 >= scale*1200 and u2 < scale*1680:
        if v1 >= scale*240 and v2 < scale*720:
            return (3, "right")

    return None  # Bounding box doesn't fit any side


class ImagePointFinder:

    def __init__(self, plotter=None, rep=None, scale=1):

        if plotter is not None:
            self.plot = plotter.getSubplot(1)
        else:
            self.plot = None

        self.reporter = rep
        self.scale = scale

        ####### CAMERA MATRIX (TRIAL)
        calib_file = "../calibration_data_intrinsics/intrinsicsUHD.pkl"
        # Open the pickle file for reading in binary mode
        with open(calib_file, 'rb') as file:
            # Load the dictionary from the pickle file
            self.camera_calib = pickle.load(file)

        ############### PREPARE THE DETECTOR
        # Load the fine-tuned model's state dictionary
        sides = ["back", "left", "right", "front", "top", "bottom"]
        # create calibration folders
        for side in sides:
            if not os.path.exists(side):
                os.makedirs(side)
                print(f"Folder '{side}' created.")
            else:
                print(f"Folder '{side}' already exists.")

        model_state_dict = torch.load("one_shot_object_detector_5x3_UHD.pth")
        model_state_dict_side = torch.load("one_shot_object_detector_5x3_UHD_SIDES.pth")

        # Create an instance of the model
        self.model = fasterrcnn_resnet50_fpn(pretrained=False)
        num_classes = 1
        self.model.roi_heads.box_predictor.cls_score.out_features = num_classes
        self.model.roi_heads.box_predictor.bbox_pred.out_features = num_classes * 4
        self.model.load_state_dict(model_state_dict)

        # Send the model to CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        # Create an instance of the model
        self.model_side = fasterrcnn_resnet50_fpn(pretrained=False)
        num_classes = 1
        self.model_side.roi_heads.box_predictor.cls_score.out_features = num_classes
        self.model_side.roi_heads.box_predictor.bbox_pred.out_features = num_classes * 4
        self.model_side.load_state_dict(model_state_dict_side)

        # Send the model to CUDA if available
        self.model_side.to(device)
        self.model_side.eval()

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.device = device
        self.scale_factor = 2

    def processImage(self, image_path, verbose=True):

        img_id = image_path.split("/")[-1]

        print("Processing image: ", image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        test_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        predictions = self.model(test_tensor)[0]

        # Find the index of the bounding box with the highest score
        # max_score_idx = predictions['scores'][:2]

        if len(predictions['boxes']) < 2:
            print("CHECKERBOARD NOT DETECT")
            if self.reporter is not None:
                self.reporter.updateFailure('chessboard_not_detected')
            return None

        # Get the bounding box and score with the highest score
        max_score_box = predictions['boxes'][:2].detach().cpu().numpy()
        # max_score = predictions['scores'][max_score_idx]

        # # Calculate the minimum x, minimum y, maximum x, and maximum y values across all bounding boxes
        # x1 = np.min(max_score_box[:, 0])
        # y1 = np.min(max_score_box[:, 1])
        # x2 = np.max(max_score_box[:, 2])
        # y2 = np.max(max_score_box[:, 3])
        #
        # # Create the merged bounding box
        # merged_bbox = [x1, y1, x2, y2]
        #
        # side1 = map_bounding_box_to_side(max_score_box[0], scale = self.scale)
        # side2 = map_bounding_box_to_side(max_score_box[1], scale = self.scale)
        #
        # if side1 == side2:
        #     side = side1
        # else:
        #     side = None
        #
        # cube = CubeProjection(Image.fromarray(image), ".")
        # img_side = cube.cube_projection(face_id=side[0], img_id="cropped_{}".format(img_id))
        #
        # img_side = np.array(img_side)
        # if verbose:
        #     plt.imshow(img_side)
        #     plt.show()

        # test_tensor = F.to_tensor(img_side).unsqueeze(0).to(self.device)
        # predictions = self.model_side(test_tensor)[0]

        # Find the index of the bounding box with the highest score
        # max_score_idx = predictions['scores'][:2]

        # if len(predictions['boxes']) < 2:
        #     print("CHECKERBOARD NOT DETECT (2nd round)")
        #     if self.reporter is not None:
        #         self.reporter.updateFailure('chessboard_not_detected_cropping')
        #     return None

        # Get the bounding box and score with the highest score
        max_score_box_side = predictions['boxes'][:2].detach().cpu().numpy()

        if max_score_box_side[0, 0] < max_score_box_side[1, 0]:
            boxes = (max_score_box_side[0], max_score_box_side[1])
        else:
            boxes = (max_score_box_side[1], max_score_box_side[0])

        box_with_corners = []  # [left/right][img, cropped_image, box, corners2]

        for box in boxes:

            # Enlarge the bounding box by a factor of 1.2 (adjust as needed)
            enlargement_factor = 1.25
            box = [
                max(0,int(np.round(
                    box[0] - (box[2] - box[0]) * (enlargement_factor - 1) / 2))),
                max(0,int(np.round(
                    box[1] - (box[3] - box[1]) * (enlargement_factor - 1) / 2))),
                max(0,int(np.round(
                    box[2] + (box[2] - box[0]) * (enlargement_factor - 1) / 2))),
                max(0,int(np.round(
                    box[3] + (box[3] - box[1]) * (enlargement_factor - 1) / 2))),
            ]

            # Crop a slice of the image using the horizontal pixel coordinates
            cropped_image = image[box[1]:box[3], box[0]:box[2]]

            # Offset for the x and y coordinates
            offset_x = box[0]
            offset_y = box[1]

            # side = map_bounding_box_to_side(box)

            ######## CALIBRATION
            if self.scale_factor > 1:
                img = cv2.resize(cropped_image, [cropped_image.shape[1] * self.scale_factor,
                                                 cropped_image.shape[0] * self.scale_factor], cv2.INTER_CUBIC)
            else:
                img = cropped_image

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (5, 3), None)
            # If found, add object points, image points (after refining them)

            print(ret)

            if not ret:
                print("stop")
            # if verbose:
            #     plt.imshow(img)
            #     plt.show()

            if ret == True:
                # objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.criteria)
                # imgpoints.append(corners2)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (5, 3), corners2, ret)

                offsets = np.zeros_like(corners2) + np.array((offset_x, offset_y))

                box_with_corners.append((img, image, box, corners2, offsets))

        if len(box_with_corners) == 2:

            # if verbose:
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(box_with_corners[0][0])
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(box_with_corners[1][0])
            #     plt.show()

            corners = np.concatenate([box_with_corners[0][3], box_with_corners[1][3]], axis=0)
            offsets = np.concatenate([box_with_corners[0][4], box_with_corners[1][4]], axis=0)

            #k = self.camera_calib[side[1]]['K']

            ##### SHIFT LEFT POINTS
            # Reshape the array to 2D (15x2)
            data = corners.reshape(-1, 2)
            offsets = offsets.reshape(-1, 2)

            data = (data / self.scale_factor) + offsets
            # Sort the array by the first column (x values)
            sort_idx = data[:, 0].argsort()
            sorted_data = data[sort_idx]

            #offsets = offsets[sort_idx]

            # Extract the first 3 points with the smallest x values
            left_x_points = sorted_data[:5]
            right_x_points = sorted_data[-5:]

            if verbose and self.plot is not None:
                self.plot.imshow(box_with_corners[0][1])
                self.plot.scatter(left_x_points[:, 0], left_x_points[:, 1], color="blue", label="Left")
                self.plot.scatter(right_x_points[:, 0], right_x_points[:, 1], color="red", label="Right")
                self.plot.legend()

            return [None, left_x_points, right_x_points]

        else:
            print("Unable to find corners, skipping...")
            if self.reporter is not None:
                self.reporter.updateFailure('corners_not_found')
            return None





class LaserPointFinder:

    def __init__(self, laser_spec, template_radious=0.420, detection_confidence=0.35, plotter=None, rep=None):


        self.angle_max = laser_spec['angle_max']
        self.angle_min = laser_spec['angle_min']
        self.angle_increment = laser_spec['angle_increment']
        self.range_min = laser_spec['range_min']
        self.range_max = laser_spec['range_max']

        if plotter is not None:
            self.plot = plotter.getSubplot(2)
        else:
            self.plot = None

        self.reporter = rep

        ##### PARAMETERS
        #################

        self.R = template_radious

        self.dist_BC = self.R * np.sqrt(2)
        self.eps_point = 15
        self.num_point = np.ceil(self.R/0.05)
        self.aperture = 15  # degrees
        self.confidence_th = detection_confidence


        template = [
            [0, 0],
            [self.R, 0],
            [0, self.R],
            [0, self.R/2],
            [self.R/2, 0],
            [0, self.R / 4],
            [self.R / 4, 0],
            [0, self.R / 2 + self.R / 4],
            [self.R / 2 + self.R / 4, 0],
        ]

        self.template = np.array(template)

    def processScan(self, ranges, verbose=True):

        ranges = np.array(ranges)
        mask = ranges <= self.range_max

        # Calculate the angles for each range measurement
        angles = np.arange(self.angle_min, self.angle_max, self.angle_increment)

        ranges = ranges[mask]
        angles = angles[mask]

        # Convert polar coordinates to Cartesian coordinates
        x = np.multiply(ranges, np.cos(angles))
        y = np.multiply(ranges, np.sin(angles))

        points = np.stack([x, y], axis=1)
        points_polar = np.stack([ranges, angles], axis=1)

        cont = 0
        selected_points = []
        selected_points_polar = []

        # Plot the laser scan data
        if verbose and self.plot is not None:
            self.plot.scatter(points[:, 0], points[:, 1], c='orange', marker='x', alpha=0.5)
            self.plot.scatter(self.template[:, 0], self.template[:, 1], c='purple', marker='1')

        # Iterate through all combinations of 3 points
        for i in tqdm(range(len(points))):

            A = points[i]

            eps = abs(np.sin(self.angle_increment) * np.linalg.norm(A))
            eps = 4 * eps

            for j in range(len(points)):
                B = points[j]
                distance_AB = np.linalg.norm(B - A)  # Euclidean distance

                distance_A = np.linalg.norm(A)
                distance_B = np.linalg.norm(B)

                if abs(distance_AB - self.R) <= eps and distance_B < distance_A:

                    count_points_on_AB = 0  # Counter for points on line AB
                    count_points_on_AC = 0  # Counter for points on line AC

                    for k in range(len(points)):

                        C = points[k]

                        distance_AC = np.linalg.norm(C - A)  # Euclidean distance
                        distance_BC = np.linalg.norm(B - C)
                        distance_C = np.linalg.norm(C)

                        #CHECK B AND C ORDER
                        flag = False
                        m = A[1] / A[0]

                        # if C[1] > m * C[0] and B[1] < m * B[0]:
                        #     flag = True


                        if A[0] < 0 and A[1] < 0:
                            if C[1] < m * C[0] and B[1] > m * B[0]:
                                flag = True
                        else:
                            if C[1] > m * C[0] and B[1] < m * B[0]:
                                flag = True

                        if flag:

                            if distance_C < distance_A and abs(distance_AC - self.R) <= eps and abs(
                                    distance_AC - distance_AB) <= eps and (distance_BC - self.dist_BC) <= eps:

                                vector_AB = B - A
                                vector_AC = C - A

                                cos_angle_AB = np.dot(vector_AB, vector_AC) / (
                                        np.linalg.norm(vector_AC) * np.linalg.norm(vector_AB))

                                cos_dist = np.arccos(np.clip(cos_angle_AB, -1.0, 1.0))

                                #if (np.degrees(cos_dist) - 90) <= 0 and abs(np.degrees(cos_dist) - 90) < self.aperture:
                                if abs(np.degrees(cos_dist) - 90) < self.aperture:

                                    # Count points on lines AB and AC

                                    # Remove points A, B, and C from filtered_points
                                    filtered_points = np.delete(points, [i, j, k], axis=0)

                                    distances = np.sqrt(np.sum((filtered_points - A) ** 2, axis=1))
                                    filtered_points = filtered_points[distances <= self.R]

                                    for point in filtered_points:

                                        vector_AB = B - A
                                        vector_AC = C - A
                                        vector_AP = point - A

                                        # Calculate the cosine of the angle between vectors
                                        cos_angle_AB = np.dot(vector_AP, vector_AB) / (
                                                np.linalg.norm(vector_AP) * np.linalg.norm(vector_AB))
                                        cos_angle_AC = np.dot(vector_AP, vector_AC) / (
                                                np.linalg.norm(vector_AP) * np.linalg.norm(vector_AC))

                                        # Calculate the angular distances (angles in radians)
                                        distance_pAB = np.arccos(np.clip(cos_angle_AB, -1.0, 1.0))
                                        distance_pAC = np.arccos(np.clip(cos_angle_AC, -1.0, 1.0))

                                        if np.degrees(distance_pAB) <= self.eps_point:
                                            count_points_on_AB += 1
                                        if np.degrees(distance_pAC) <= self.eps_point:
                                            count_points_on_AC += 1

                                    if count_points_on_AB > self.num_point and count_points_on_AC > self.num_point:

                                        if verbose and self.plot is not None:
                                            self.plot.scatter(A[0], A[1], c='cyan', marker='+', s=400, linewidths=3,  alpha=0.25)
                                            self.plot.scatter(B[0], B[1], c='red', marker='+', s=400, linewidths=3,  alpha=0.25)
                                            self.plot.scatter(C[0], C[1], c='blue', marker='+', s=400, linewidths=3,  alpha=0.25)

                                        cont += 1
                                        selected_points.append({'A': A, "B": B, "C": C})
                                        selected_points_polar.append({"A":points_polar[i], "B":points_polar[j], "C":points_polar[k]})

        print("NUMBER OF PATTERNS FOUND: ", cont)

        if cont == 0:
            if verbose and self.plot is not None:
                self.plot.cla()

            if self.reporter is not None:
                self.reporter.updateFailure('not_enough_laser_patterns')
            return None, None

        if verbose and self.plot is not None:
            self.plot.set_xlabel('X Distance (m)')
            self.plot.set_ylabel('Y Distance (m)')
            self.plot.axis('equal')  # Equal aspect ratio
            self.plot.grid(True)

        final_point = []
        final_point_polar = []
        min_dist = 10000

        for idx, center in enumerate(selected_points):

            A = center['A']
            B = center['B']
            C = center['C']

            distances = np.sqrt(np.sum((points - A) ** 2, axis=1))
            window_points = points[distances <= 1.5 * self.R]

            theta = np.arctan2(B[1] - A[1], B[0] - A[0])

            # Apply the sliding window to the point cloud
            optimal_translation, optimal_rotation, result = self.template_matching(window_points, self.template,
                                                                              initial_params=np.array(
                                                                                  [A[0], A[1], theta]))

            # Calculate the distance from the result
            distance = result.fun

            if distance < min_dist:
                min_dist = distance
                final_point = (A, B, C)
                final_point_polar = (selected_points_polar[idx]["A"], selected_points_polar[idx]["B"], selected_points_polar[idx]["C"])

        A, B, C = final_point

        #print("DETECTED POINTS: \nA({}, {}) \nB({}, {}), \nC({}, {})".format(A[0], A[1], B[0], B[1], C[0], C[1] ))
        print("MIN DIST: ", min_dist)


        if verbose and self.plot is not None:
            self.plot.scatter(A[0], A[1], c='cyan', marker='o', s=200, linewidths=3, label="Final prediction A", alpha=0.5)
            self.plot.scatter(B[0], B[1], c='red', marker='o', s=200, linewidths=3, label="Final prediction B", alpha=0.5)
            self.plot.scatter(C[0], C[1], c='blue', marker='o', s=200, linewidths=3, label="Final prediction C", alpha=0.5)

            self.plot.legend()

            # Convert Cartesian to polar coordinates
            cartesian_points = np.array(final_point)
            #polar_coordinates = cartesian_to_polar(cartesian_points)
            self.plot.axis('equal')  # Equal aspect ratio


        if min_dist < self.confidence_th:
            return final_point, final_point_polar
        else:
            print("Confidence is too low! Discard the detection...")
            if self.reporter is not None:
                self.reporter.updateFailure('low_confidence_laser_pattern')
            return None, None

    def template_matching(self, point_cloud, template, initial_params=np.array([0, 0, 0])):
        # Define a function to minimize - the sum of distances
        def objective(params):
            translation = params[:2]
            rotation = params[2]

            # Rotate the template by the given angle
            rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                                        [np.sin(rotation), np.cos(rotation)]])
            rotated_template = np.dot(template, rotation_matrix.T)

            # Translate the rotated template
            translated_template = rotated_template + translation

            # Calculate distances between translated template and point cloud
            distances = cdist(translated_template, point_cloud)

            # Return the negative sum of distances (to be minimized)
            return np.mean(distances)

        # Initial guess for parameters: [translation_x, translation_y, rotation]

        # Minimize the objective function
        result = minimize(objective, initial_params, method='Nelder-Mead')

        # Get the optimal translation and rotation values
        optimal_translation = result.x[:2]
        optimal_rotation = result.x[2]

        return optimal_translation, optimal_rotation, result


def main():

    verbose = True

    image_folder = './imagesUHD_board_static_i'
    csv_file_path = os.path.join(image_folder,'scan.csv')

    plotter = Plotter((1,2))
    reporter = Reporter()

    # Define laser specification
    laser_spec = {
        'frame_id': "base_link",
        'angle_min': -3.140000104904175,
        'angle_max': 3.140000104904175,
        'angle_increment': 0.005799999926239252,
        'range_min': 0.44999998807907104,
        'range_max': 25.0
    }

    img_processor = ImagePointFinder(plotter=plotter, rep=reporter, scale = 2)
    laser_processor = LaserPointFinder(laser_spec, plotter=plotter, rep=reporter)

    good_points = []

    with open(csv_file_path, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip the header row

        for i, row in enumerate(csvreader):

            # if i<8: #FOR TESTING
            #     continue

            bag_name = row[0]
            laser_ranges = [float(value) for value in row[1:]]

            #Prepare the related image
            image_filename = os.path.splitext(bag_name)[0] + '.png'
            image_path = os.path.join(image_folder, image_filename)

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            img_results = img_processor.processImage(image_path, verbose=verbose)

            if img_results is not None:

                side, left, right = img_results
                print("DETECTED POINTS in side {}: ".format(side))
                # Iterate through the points and print them
                for j, point in enumerate(left, 1):
                    x, y = point
                    print(f"LEFT {j}: ({x}, {y})")
                for j, point in enumerate(right, 1):
                    x, y = point
                    print(f"RIGHT {j}: ({x}, {y})")

                laser_results, laser_results_polar = laser_processor.processScan(laser_ranges, verbose=verbose)

                if laser_results is not None:

                    A,B,C = laser_results
                    Ap,Bp,Cp = laser_results_polar
                    print("DETECTED POINTS: \nA({}, {}) \nB({}, {}), \nC({}, {})".format(A[0], A[1], B[0], B[1], C[0], C[1] ))
                    print("DETECTED POINTS (POLAR): \nA({}, {}) \nB({}, {}), \nC({}, {})".format(Ap[0], Ap[1], Bp[0], Bp[1], Cp[0], Cp[1] ))
                    print(f"CHECK polar2cartesian: \nA({Ap[0] * np.cos(Ap[1])}, {Ap[0] * np.sin(Ap[1])}) \nB({Bp[0] * np.cos(Bp[1])}, {Bp[0] * np.sin(Bp[1])}), \nC({Cp[0] * np.cos(Cp[1])}, {Cp[0] * np.sin(Cp[1])})")

                    print("!!!! SUCCESSFULLY DETECTED POINTS")
                    if verbose:
                        #plotter.show()
                        plotter.zoom_around_point(A[0], A[1], 4.0)
                        plotter.save("pano_process", os.path.splitext(bag_name)[0], subfolder="indoor_static")

                    #DATA ASSOCIATION AND SAVE: left image point = C, right image points = B
                    data = ((C, left), (B, right), image_filename, (Cp, left), (Bp, right) )
                    good_points.append(data)

            plotter.clear()

            if i%50 == 0:

                # Specify the file path where you want to save the dictionary
                file_path = "cameraLaser_pointsUHD_static_indoor.pkl"
                # save dictionary to pkl file
                with open(file_path, 'wb') as fp:
                    pickle.dump(good_points, fp)
                    print('Temporary saved to {}'.format(file_path))

                reporter.printReport('failures')

            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print(f"Total tuples detected: {len(good_points)}")

    # Specify the file path where you want to save the dictionary
    file_path = "cameraLaser_pointsUHD_static_indoor.pkl"

    # save dictionary to pkl file
    with open(file_path, 'wb') as fp:
        pickle.dump(good_points, fp)
        print('List saved successfully to {}'.format(file_path))


    reporter.printReport('failures')

    print("***************** PROCESSING ENDS")



if __name__ == "__main__":

    main()