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
from deep_sort_realtime.deepsort_tracker import DeepSort

import detect_people
from src.auto_labeling_tools.util.cube_projection import from_cube2panoramic
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
    yolo_model = detect_people.load_model(weights="models/yolov7.pt")

    # set tracker
    tracker = DeepSort(max_age=15)

    for i in range(30, len(ids)):
        scan = scans[i]
        id = ids[i]

        img_path = osp.join(path_images, str(id).zfill(4)+".png")
        print(img_path)

        # read panoramic
        img1 = cv2.imread(img_path)
        cv2_image_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rep_image = cv2_image_rgb.copy()
        rep_image_track = cv2_image_rgb.copy()
        cropped = cv2_image_rgb[576:, :].copy()

        # project to side
        sides = CubeProjection(Image.fromarray(cv2_image_rgb), '')
        sides.cube_projection()

        # for each face
        rep_detections = []
        face_detections = {}
        inside_detections = []
        border_detections = []
        for face, side_img in sides.sides.items():
            if face in ["front", "back", "left", "right"]:

                face_det = []
                # detect
                cv_image = np.array(side_img)
                detected, objects_pose = detect_people.detect_person(cv_image, yolo_model)
                face_detections[face] = np.array(detected)
                if len(detected) > 0:
                    border_det = list((np.array(detected)[:, 0] > 5) & (np.array(detected)[:, 2] < 955))
                    removed_elements = 0

                    # reproject detections
                    for j in range(len(detected)):
                        pan_point_1 = from_cube2panoramic(face, [detected[j][0], detected[j][1]])
                        pan_point_2 = from_cube2panoramic(face, [detected[j][2], detected[j][3]])

                        # check if it cross the edge
                        if abs(pan_point_2[0] - pan_point_1[0]) < image_spec["width"]/2:
                            rep_detections.append([pan_point_1[0], pan_point_1[1], pan_point_2[0], pan_point_2[1]])
                        else:
                            del border_det[j - removed_elements]
                            removed_elements += 1

                    inside_detections += border_det

                # plot side
                plot_detection(cv_image, detected, face+str(id).zfill(4), out_path_det+"/side/"+face+"_")

        # plot only border
        outside_detections = [not d for d in inside_detections]
        rep_inside = np.array(rep_detections)[inside_detections]
        for det in rep_inside:
            rep_image = cv2.rectangle(rep_image, det[:2], det[2:], (0, 255, 0), 5)

        # procedure to merge bordere detections
        rep_outside = np.array(rep_detections)[outside_detections]
        merged = [False for j in range(len(rep_outside))]
        new_detections = []
        for k1 in range(len(rep_outside)-1):
            for k2 in range(k1+1, len(rep_outside)):
                # check if to merge
                if not merged[k1]:
                    if 0 < (rep_outside[k2][0] - rep_outside[k1][2]) < 20:
                        # merge
                        new_detections.append([rep_outside[k1][0], min(rep_outside[k1][1], rep_outside[k2][1]), rep_outside[k2][2], max(rep_outside[k1][3], rep_outside[k2][3])])
                        merged[k1] = True
                        merged[k2] = True
                        continue
                    elif 0 < (rep_outside[k1][0] - rep_outside[k2][2]) < 20:

                        new_detections.append(
                            [rep_outside[k2][0], min(rep_outside[k1][1], rep_outside[k2][1]), rep_outside[k1][2], max(rep_outside[k1][3], rep_outside[k2][3])])
                        merged[k1] = True
                        merged[k2] = True
                        continue
            if not merged[k1]:
                new_detections.append(rep_outside[k1])

        if len(merged) > 0:
            if not merged[-1]:
                new_detections.append(rep_outside[-1])

        for det in new_detections:
            rep_image = cv2.rectangle(rep_image, det[:2], det[2:], (255, 0, 0), 5)

        if len(new_detections) > 0:
            concatenated = np.concatenate((new_detections, rep_inside))
        else:
            concatenated = rep_inside
        # plot reprojected
        cv2.imwrite(out_path_det+'/rep/rep_' + str(str(id).zfill(4)) + '.jpg', cv2.cvtColor(rep_image, cv2.COLOR_BGR2RGB))

        #detected_pan, objects_poses = detect_people.detect_person(cv2_image_rgb, yolo_model)
        #plot_detection(cv2_image_rgb, detected_pan, str(id).zfill(4), out_path_det+"/pan/")

        # detected_c, _ = detect_people.detect_person(cropped, yolo_model)
        # plot_detection(cropped, detected_c, str(id).zfill(4), out_path_det + "/cropped/")

        # at this point
        # yolo from panoramic = detected_pan
        # yolo from sides = face_detections
        # yolo reprojected panoramic = rep_detections

        rep_detections = np.array(concatenated)
        rep_detections[:, 2] = rep_detections[:, 2] - rep_detections[:, 0]
        rep_detections[:, 3] = rep_detections[:, 3] - rep_detections[:, 1]

        raw_det = []
        for det in rep_detections:
            raw_det.append((det, 1, 0))

        tracks = tracker.update_tracks(raw_det, frame=cv2_image_rgb)
        # tracks = tracker.update_tracks(bbs, frame=frame)  # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            rep_image_track = cv2.rectangle(rep_image_track, ltrb[:2].astype(int), ltrb[2:].astype(int), (0, 255, 0), 5)
            rep_image_track = cv2.putText(rep_image_track, track_id, ltrb[:2].astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imwrite(out_path_det+'/track/track_' + str(str(id).zfill(4)) + '.jpg', cv2.cvtColor(rep_image_track, cv2.COLOR_BGR2RGB))
