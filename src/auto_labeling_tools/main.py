import csv
import os
import os.path as osp
import pickle

import cv2
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')

from matplotlib import pyplot as plt, patches

import detect_people
from src.auto_calibration_tools.scripts.camera_laser_calibration.calibrateCamera2LaserPnP import projectPoint2Image
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
    path_out = "/home/leonardo/Downloads/lab_indoor_1/" #read processed bag
    out_path_scans = "/home/leonardo/Downloads/lab_indoor_1/img_out" #visualization
    out_path_det = "/home/leonardo/Downloads/lab_indoor_1/yolo_out" #visualization
    out_path_associations = "/home/leonardo/Downloads/lab_indoor_1/a" #visualization
    out_path_annotations = "/home/leonardo/Downloads/lab_indoor_1/"

    # Specify the path of the calibration data
    calibration_path = "../auto_calibration_tools/scripts/camera_laser_calibration/laser2camera_map.pkl"

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

    # ouput path
    out_file = os.path.join(out_path_annotations, "automatic_annotations.csv")
    if os.path.exists(out_file):
        file_annotation = open(out_file, "a")
    else:
        file_annotation = open(out_file, "w")

    # read scans
    scans, ids = read_scan(path_laser)

    # Load calibration
    with open(calibration_path, 'rb') as file:
        calibration_map = pickle.load(file)
    H = calibration_map['H2']

    # set yolo
    #yolo_model = detect_people.load_model(weights="models/yolov7.pt")
    imde = ImageDetector("models/yolov7.pt")

    # set laser detector
    ld = LaserDetector(scans, laser_spec)

    for i in range(30, len(ids)):
        scan = scans[i]
        id = ids[i]
        image_index = str(id).zfill(4)

        img_path = osp.join(path_images, image_index+".png")
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
        rep_detections, face_detections, _ = imde.compute_detections_sides(cv_image, out_path_det, frame=i)

        # at this point
        # yolo from panoramic = detected_pan
        # yolo from sides = face_detections
        # yolo reprojected panoramic = rep_detections
        # point detected as people from laser = people
        # label of each point = labels
        # centroid of each detected person = cluster_centroids

        lines = projectPoint2Image(cluster_centroids, H)
        distances = np.linalg.norm(cluster_centroids, axis=1)

        '''
        # visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # Plot the image in the first subplot
        ax1.imshow(cv_image)
        ax1.set_title(f"Image {str(id).zfill(4)}")
        ax1.axis('off')  # Turn off axis labels and ticks

        # Plot the laser scan data in the second subplot
        ax2.scatter(points[:, 0], points[:, 1], marker="o", color="orange", s=2)
        ax2.set_title("Laser Scan Data")
        ax2.axis('equal')

        colors = mpl.cm.rainbow(np.linspace(0, 1, len(lines)))
        ax2.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], marker="x", color=colors, s=3)
        for j in range(len(lines)):
            ax1.axvline(lines[j], c=colors[j])

        plt.show()


        '''

        box_init = rep_detections[:, 0]
        box_end = rep_detections[:, 2]

        # create assosiation matrix
        assosiations = np.zeros((len(lines), len(rep_detections)))
        for k, line in enumerate(lines):
            assosiation = (box_init < line) & (box_end > line)
            assosiations[k] = assosiation.astype(int)

        # resolve more lasers for a detection
        for col_index in np.where((np.count_nonzero(assosiations, axis=0) > 1))[0]:
            col = assosiations[:, col_index]
            selected_index = np.argmax((1/distances * col))
            assosiations[:, col_index] = 0
            assosiations[selected_index, col_index] = 1

        # resolve for one laser more boxes
        assosiations[np.where((np.count_nonzero(assosiations, axis=1) > 1))] = 0

        # results
        centroids, boxes = np.where(assosiations==1)

        # visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # Plot the image in the first subplot
        ax1.imshow(cv_image)
        ax1.set_title(f"Image {str(id).zfill(4)}")
        ax1.axis('off')  # Turn off axis labels and ticks
        # Plot the laser scan data in the second subplot
        ax2.scatter(points[:, 0], points[:, 1], marker="o", color="black", s=2)
        ax2.set_title("Laser Scan Data")
        ax2.axis('equal')

        colors = mpl.cm.rainbow(np.linspace(0, 1, len(centroids)))
        ax2.scatter(cluster_centroids[centroids, 0], cluster_centroids[centroids, 1], marker="x", color=colors, s=6)
        for j, line_number in enumerate(centroids):
            ax1.axvline(lines[line_number], c=colors[j])

        for j, det in enumerate(boxes):
            box = rep_detections[det].copy()
            # Create a rectangle patch using the coordinates and dimensions
            top_left = box[:2]
            bottom_right = box[2:]

            width = bottom_right[0] - top_left[0]
            height = top_left[1] - bottom_right[1]

            top_left[1] = bottom_right[1]

            rectangle = patches.Rectangle(top_left, width, height, linewidth=1, edgecolor=colors[j], facecolor='none')

            # Add the rectangle patch to the axis
            ax1.add_patch(rectangle)

        #rep_image = cv2.rectangle(rep_image, det[:2], det[2:], (255, 0, 0), 5)

        plt.savefig(osp.join(out_path_associations, image_index+".png"))
        plt.close()

        file_annotation.write(image_index)
        for e in range(len(centroids)):
            box = rep_detections[boxes[e]]
            centroid = cluster_centroids[centroids[e]]
            file_annotation.write(";" + str(box[0]) + ";" + str(box[1]) + ";" + str(box[2]) + ";" + str(box[3]))
            file_annotation.write(";" + str(centroid[0]) + ";" + str(centroid[1]))
        file_annotation.write("\n")
        file_annotation.flush()
        os.fsync(file_annotation.fileno())

    file_annotation.close()

