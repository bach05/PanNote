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

from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

import matplotlib.patches as patches

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
                'ball_not_detected': 0,
                'no_circle_detected': 0,
                'unable_to_side_remap':0,
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

    side_x_offset = 240 * scale  # X offset for each side
    side_y_offset = 240 * scale  # Y offset for each side
    side_width = 480 * scale  # Width of each side
    side_height = 480 * scale  # Height of each side

    if (u1 >= 0 and u2 < scale*240) or (u1 >= scale*1680 and u2 < scale*1920):
        if v1 >= scale*240 and v2 < scale*720:
            side = (0, "back")
        else:
            return None, None  # Bounding box doesn't fit any side
    elif u1 >= scale*240 and u2 < scale*720:
        if v1 >= scale*240 and v2 < scale*720:
            side = (1, "left")
        else:
            return None, None  # Bounding box doesn't fit any side
    elif u1 >= scale*720 and u2 < scale*1200:
        if v1 >= scale*240 and v2 < scale*720:
            side = (2, "front")
        else:
            return None, None  # Bounding box doesn't fit any side
    elif u1 >= scale*1200 and u2 < scale*1680:
        if v1 >= scale*240 and v2 < scale*720:
            side = (3, "right")
        else:
            return None, None  # Bounding box doesn't fit any side
    else:
        return None, None  # Bounding box doesn't fit any side

    # Calculate the coordinates of the bounding box within the detected side
    x1 = max(u1 - side_x_offset, 0)
    y1 = max(v1 - side_y_offset, 0)
    x2 = min(u2 - side_x_offset, side_width)
    y2 = min(v2 - side_y_offset, side_height)

    return side, (x1, y1, x2, y2)


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

        model_state_dict = torch.load("one_shot_ball_detector.pth")
        model_state_dict_side = torch.load("one_shot_ball_detector_SIDES.pth")

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

        if len(predictions['boxes']) < 1:
            print("BALL NOT DETECT")
            if self.reporter is not None:
                self.reporter.updateFailure('ball_not_detected')
            return None

        # Get the bounding box and score with the highest score
        max_score_box = predictions['boxes'][0].detach().cpu().numpy()
        # max_score = predictions['scores'][max_score_idx]

        # fig, ax = plt.subplots()
        # ax.imshow(image)
        # ax.axvline(240*self.scale, color="white")
        # ax.axvline(720*self.scale, color="white")
        # ax.axvline(1200*self.scale, color="white")
        # ax.axvline(1680*self.scale, color="white")
        #
        # ax.axhline(240*self.scale, color="white")
        # ax.axhline(720*self.scale, color="white")
        #
        # x_min, y_min, x_max, y_max = max_score_box
        # box = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2,
        #                         edgecolor='r',
        #                         facecolor='none')
        # ax.add_patch(box)
        # plt.show()

        side, max_score_box = map_bounding_box_to_side(max_score_box, scale=self.scale)

        if side is not None:

            cube = CubeProjection(Image.fromarray(image), ".")
            img_side = cube.cube_projection(face_id=side[0], img_id="cropped_{}".format(img_id))

            test_tensor = F.to_tensor(img_side).unsqueeze(0).to(self.device)
            predictions = self.model_side(test_tensor)[0]
            if len(predictions['boxes']) < 1:
                print("BALL NOT DETECT (2)")
                if self.reporter is not None:
                    self.reporter.updateFailure('ball_not_detected')
                return None

            max_score_box = predictions['boxes'][0].detach().cpu().numpy()

            x1, y1, x2, y2 = max_score_box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # fig, ax = plt.subplots()
            # ax.imshow(img_side)
            # x_min, y_min, x_max, y_max = max_score_box
            # box = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2,
            #                         edgecolor='r',
            #                         facecolor='none')
            # ax.add_patch(box)
            # ax.scatter(center_x, center_y, marker='+', color="r")
            # plt.show()

            if verbose and self.plot is not None:
                self.plot.imshow(img_side)
                x_min, y_min, x_max, y_max = max_score_box
                box = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2,
                                        edgecolor='r',
                                        facecolor='none')
                self.plot.add_patch(box)
                self.plot.scatter(center_x, center_y, marker='+', color="r")

            return [side, center_x, center_y]

        else:
            print("Unable to map the image to one side")
            if self.reporter is not None:
                self.reporter.updateFailure('unable_to_side_remap')
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

        # self.R = template_radious
        #
        # self.dist_BC = self.R * np.sqrt(2)
        # self.eps_point = 15
        # self.aperture = 15  # degrees
        # self.confidence_th = detection_confidence

        self.diameter = 0.7
        self.num_point = 10
        self.radius = 0.65/2


        # template = [
        #     [0, 0],
        #     [self.R, 0],
        #     [0, self.R],
        #     [0, self.R/2],
        #     [self.R/2, 0],
        #     [0, self.R / 4],
        #     [self.R / 4, 0],
        #     [0, self.R / 2 + self.R / 4],
        #     [self.R / 2 + self.R / 4, 0],
        # ]

        # self.template = np.array(template)

    def circle_from_points(self, point1, point2, point3):
        # Extract coordinates of the three points
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3

        # Calculate the midpoints of the lines between the points
        mid_x1 = (x1 + x2) / 2
        mid_y1 = (y1 + y2) / 2
        mid_x2 = (x2 + x3) / 2
        mid_y2 = (y2 + y3) / 2

        # Calculate the slopes of the lines perpendicular to the segments
        if x2 - x1 == 0:
            slope1 = None  # Vertical line, slope is undefined
        else:
            slope1 = -(y2 - y1) / (x2 - x1)
        if x3 - x2 == 0:
            slope2 = None  # Vertical line, slope is undefined
        else:
            slope2 = -(y3 - y2) / (x3 - x2)

        # Check if the slopes are zero or infinite (vertical or horizontal lines)
        if slope1 is None:
            center_x = mid_x1
            center_y = slope2 * (center_x - mid_x2) + mid_y2
        elif slope2 is None:
            center_x = mid_x2
            center_y = slope1 * (center_x - mid_x1) + mid_y1
        elif slope1 == 0:
            center_y = mid_y1
            center_x = (center_y - mid_y2) / slope2 + mid_x2
        elif slope2 == 0:
            center_y = mid_y2
            center_x = (center_y - mid_y1) / slope1 + mid_x1
        else:
            # Calculate the center of the circle
            center_x = (slope1 * mid_x1 - slope2 * mid_x2 + mid_y2 - mid_y1) / (slope1 - slope2)
            center_y = slope1 * (center_x - mid_x1) + mid_y1

        # Calculate the radius of the circle
        radius = np.sqrt((center_x - x1) ** 2 + (center_y - y1) ** 2)

        # Write the equation of the circle in standard form: (x - h)^2 + (y - k)^2 = r^2
        h = center_x
        k = center_y
        equation = f"(x - {h})^2 + (y - {k})^2 = {radius ** 2}"

        return h,k,r

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
            # self.plot.scatter(self.template[:, 0], self.template[:, 1], c='purple', marker='1')


        cluod_image, ps = self.point_cloud_to_image(points, laser_range=np.max(ranges), resolution=500)

        rows = cluod_image.shape[0]
        circles = cv2.HoughCircles(cluod_image, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                  param1=10, param2=6,
                                  minRadius=3, maxRadius=10)


        if circles is not None:
            circles = np.uint16(np.around(circles))
            detected_circles = []

            for i in circles[0, :]:
                center = (i[0], i[1])
                # # circle center
                # cv2.circle(cluod_image, center, 1, (0, 100, 100), 3)
                # # circle outline
                radius = i[2]
                # cv2.circle(cluod_image, center, radius, (255, 0, 255), 3)

                #analyze each circle
                x_c = i[0]
                y_c = i[1]
                r = radius
                margin = 1.25
                x_min = x_c - r*margin
                x_max = x_c + r*margin
                y_min = y_c - r*margin
                y_max = y_c + r*margin
                bbox = (x_min, x_max, y_min, y_max)

                # # Create a figure and axis
                # fig, ax = plt.subplots()
                #
                # ax.imshow(cluod_image)
                #
                # # Plot the bounding box as a rectangle
                # box = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r',
                #                          facecolor='none')
                # ax.add_patch(box)
                #
                # # Plot the circle
                # circle = patches.Circle((x_c, y_c), radius, fill=False, color='yellow')
                # ax.add_patch(circle)
                #
                # plt.show()

                roi = self.filter_point_cloud_by_bbox(points, bbox)

                if roi.shape[0] > self.num_point:

                    self.plot.scatter(roi[:, 0], roi[:, 1], c='red', marker='x', alpha=0.75)

                    initial_guess = [np.mean(roi[:, 0]), np.mean(roi[:, 1]), self.diameter/2]
                    (center_x, center_y, radius), cost = self.fit_circle(roi, initial_guess)

                    #print("FOUND A CIRCLE TARGET IN THE POINT CLOUD with {} score".format(cost))


                    # Check if the radius is smaller than R
                    if 0.15 < radius < self.diameter/2 and cost < 0.005:
                        print(f"Detected a GOOD ARC in the circle xc:{center_x}, yc:{center_y} r:{radius}")

                        # Create a figure and axis
                        # fig, ax = plt.subplots()
                        # ax.imshow(cluod_image)
                        # # Plot the bounding box as a rectangle
                        # box = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2,
                        #                         edgecolor='r',
                        #                         facecolor='none')
                        # ax.add_patch(box)
                        # # Plot the circle
                        # circle = patches.Circle((x_c, y_c), radius, fill=False, color='yellow')
                        # ax.add_patch(circle)
                        # plt.show()

                        circle_params = (center_x, center_y, radius, cost, roi.shape[0])
                        detected_circles.append(circle_params)

            if len(detected_circles) > 0:

                detected_circles = np.array(detected_circles)
                index = np.argmin(detected_circles[:, 3])
                best_circle = detected_circles[index, :]
                xc, yc, radius, _, count = best_circle

                radius = min(self.radius, radius)

                d = np.sqrt( self.radius**2 - radius**2)
                zc = d

                if np.isnan(zc):
                    print("################ ZC is nan")

                print(f"CIRCLE DETECTION: xc {xc}, yc {yc}, r {radius}, cont {count}")

                circle = patches.Circle((xc, yc), radius, fill=False, color='blue')
                self.plot.add_patch(circle)
                self.plot.set_aspect('equal')

                return (xc, yc, zc)

            else:
                print("NO CIRCLE DETECTED")
                if self.reporter is not None:
                    self.reporter.updateFailure('no_circle_detected')
                return None


        else:
            print("NO CIRCLES FOUND")
            if self.reporter is not None:
                self.reporter.updateFailure('no_circle_detected')
            return None


    def fit_circle(self, points, initial_guess):
        def circle_residuals(params, points):
            x_c, y_c, r = params
            distances = np.sqrt((points[:, 0] - x_c) ** 2 + (points[:, 1] - y_c) ** 2)
            return distances - r

        result = least_squares(circle_residuals, initial_guess, args=(points,))
        return result.x, result.cost # Returns (x_c, y_c, radius)

    def filter_point_cloud_by_bbox(self, points, bounding_box):

        filtered_points = []

        # Extract bounding box coordinates
        x1, x2, y1, y2 = bounding_box

        # Scale factor to map points to image coordinates
        scale = self.resolution / (2 * self.laser_range)

        # Translate points to the center of the image
        translation = np.array([self.resolution / 2, self.resolution / 2])

        for point in points:
            # Scale and translate the point to image coordinates
            x, y = point[0] * scale + translation[0], point[1] * scale + translation[1]

            # Check if the point is inside the bounding box
            if x1 <= x <= x2 and y1 <= y <= y2:
                filtered_points.append(point)

        return np.array(filtered_points)

    def point_cloud_to_image(self, points, laser_range, resolution):

        self.resolution = resolution
        self.laser_range = laser_range

        # Create an empty image with the specified resolution
        image = np.zeros((resolution, resolution), dtype=np.uint8)

        # Scale factor to map points to image coordinates
        scale = resolution / (2 * laser_range)
        pixel_size = 1 / scale

        # Translate points to the center of the image
        translation = np.array([resolution / 2, resolution / 2])

        for point in points:
            # Scale and translate the point to image coordinates
            x, y = point[0] * scale + translation[0], point[1] * scale + translation[1]

            # Ensure the point is within the image bounds
            if 0 <= x < resolution and 0 <= y < resolution:
                # Set the pixel value to 255 (white) at the point's coordinates
                image[int(y), int(x)] = 255

        return image, scale

    def belongToCircle(self, A, h, k, r, eps):
        x, y = A
        err =  (x - h)**2 + (y - k)**2 - r**2

        if err < eps:
            return True
        else:
            return False

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

    image_folder = './imagesUHD_ball_400o'
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


    good_points = {
        "back": [],
        "left": [],
        "right": [],
        "front": []
    }

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

                side, x_center, y_center = img_results
                print("DETECTED CENTER in {}, {}: ".format(x_center, y_center))

                laser_results = laser_processor.processScan(laser_ranges, verbose=verbose)

                if laser_results is not None:

                    xc, yc, zc = laser_results
                    print(f"SPHERE CENTER: xc {xc}, yc {yc}, zc {zc}")

                    print("!!!! SUCCESSFULLY DETECTED POINTS")
                    if verbose:
                        #plotter.show()
                        #plotter.zoom_around_point(xc, yc, 4.0)
                        plotter.save("pano_process", os.path.splitext(bag_name)[0], subfolder="ball_sides_o")

                    #DATA ASSOCIATION AND SAVE: left image point = C, right image points = B
                    data = (image_filename, (x_center, y_center), (xc, yc, zc) )
                    good_points[side[1]].append(data)

            plotter.clear()

            if i%50 == 0:

                # Specify the file path where you want to save the dictionary
                file_path = "cameraLaser_pointsUHD_ball_sides_o.pkl"
                # save dictionary to pkl file
                with open(file_path, 'wb') as fp:
                    pickle.dump(good_points, fp)
                    print('Temporary saved to {}'.format(file_path))

                reporter.printReport('failures')

            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    for key, value in good_points.items():
        print(f"[{key}] Total tuples detected: {len(value)}")

    # Specify the file path where you want to save the dictionary
    file_path = "cameraLaser_pointsUHD_ball_sides_o.pkl"

    # save dictionary to pkl file
    with open(file_path, 'wb') as fp:
        pickle.dump(good_points, fp)
        print('List saved successfully to {}'.format(file_path))


    reporter.printReport('failures')

    print("***************** PROCESSING ENDS")




if __name__ == "__main__":

    main()