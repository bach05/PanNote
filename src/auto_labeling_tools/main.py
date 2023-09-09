import csv
import os
import os.path as osp

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
from math import pi, atan2, hypot, floor
from numpy import clip

import detect_people
from src.auto_labeling_tools.util.cube_projection import from_cube2panoramic
from src.auto_labeling_tools.util.image_detector import ImageDetector
from src.auto_labeling_tools.util.laser_detector import LaserDetector
from src.auto_labeling_tools.util.visualization import plot_scans, plot_detection
from src.auto_labeling_tools.util.cube_projection import CubeProjection


def read_scan(path):
    scan = []
    with open(path, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Read each row of the CSV file
        image_ids = []
        for row in reader:
            image_id = int(row[0])  # Extract the image ID from the first column
            ranges = [float(value) for value in row[1].split(",")]  # Extract th
            scan.append(ranges)
            image_ids.append(image_id)

    return np.array(scan), image_ids


if __name__ == "__main__":

    #path_bag = "/media/leonardo/Elements/bag/hospital1_static.bag"
    # t_laser = "/scan"
    # t_img = "/theta_camera/image_raw"
    path_out = "/media/leonardo/Elements/prova" #read processed bag
    out_path_scans = "/media/leonardo/Elements/prova/img_out" #visualization
    out_path_det = "/media/leonardo/Elements/prova/yolo_out" #visualization

    laser_spec = {
        'frame_id': "base_link",
        'angle_min': -3.140000104904175,
        'angle_max': 3.140000104904175,
        'angle_increment': 0.005799999926239252,
        'range_min': 0.44999998807907104,
        'range_max': 25.0,
        'len': 1083
    }

    image_spec = {
        "width" : 3840,
    }

    path_laser = osp.join(path_out, "laser.csv")
    path_images = osp.join(path_out, "img")

    # read scans
    scans, ids = read_scan(path_laser)

    # set yolo
    #yolo_model = detect_people.load_model(weights="models/yolov7.pt")
    imde = ImageDetector("models/yolov7.pt")

    # set laser detector
    ld = LaserDetector(scans, laser_spec)

    for i in range(len(ids)):
        scan = scans[i]
        id = ids[i]

        img_path = osp.join(path_images, str(id).zfill(4)+".png")
        print(img_path)

        # detect people from laser
        people, labels, points, cluster_centroids = ld.detect_people(scan)
        if len(labels) > 0:
            plot_scans(points, people, out_path_scans, str(id).zfill(4), labels)
            plot_scans(points, cluster_centroids, out_path_scans, str(id+10000).zfill(4))

        # read panoramic
        img1 = cv2.imread(img_path)
        cv2_image_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        cv_image = np.array(cv2_image_rgb)
        rep_detections, face_detections, _ = imde.compute_detections_sides(cv_image)

        # at this point
        # yolo from panoramic = detected_pan
        # yolo from sides = face_detections
        # yolo reprojected panoramic = rep_detections
        # point detected as people from laser = people
        # label of each point = labels
        # centroid of each detected person = cluster_centroids

