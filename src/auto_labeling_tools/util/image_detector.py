import csv
import os
import os.path as osp

import time
import torch
from numpy import random
import cv2
import numpy as np
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.auto_labeling_tools.util.cube_projection import from_cube2panoramic
from src.auto_labeling_tools.util.laser_detector import LaserDetector
from src.auto_labeling_tools.util.visualization import plot_scans, plot_detection
from src.auto_labeling_tools.util.cube_projection import CubeProjection

from src.auto_labeling_tools.yolov7.models.experimental import attempt_load
from src.auto_labeling_tools.yolov7.utils.datasets import letterbox, LoadImages
from src.auto_labeling_tools.yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging, xyn2xy
from src.auto_labeling_tools.yolov7.utils.plots import plot_one_box
from src.auto_labeling_tools.yolov7.utils.torch_utils import select_device, time_synchronized, load_classifier, TracedModel

class ImageDetector:
    def __init__(self, model_path="../models/yolov7.pt", tracking=False, width=3840):
        self.yolo_model = load_model(weights=model_path)
        self.tracking = tracking
        self.width = width
        if self.tracking:
            self.tracker = DeepSort(max_age=15)

    def compute_detections_sides(self, img, out_path_det=None, frame=0):
        if out_path_det is not None:
            rep_image = img.copy()
            rep_image_track = img.copy()

        sides = CubeProjection(Image.fromarray(img), '')
        sides.cube_projection()

        # for each face
        rep_detections = []
        face_detections = {}
        inside_detections = []
        for face, side_img in sides.sides.items():
            if face in ["front", "back", "left", "right"]:

                face_det = []
                # detect
                cv_image = np.array(side_img)
                detected, objects_pose = detect_person(cv_image, self.yolo_model)
                face_detections[face] = np.array(detected)
                if len(detected) > 0:
                    border_det = list((np.array(detected)[:, 0] > 5) & (np.array(detected)[:, 2] < 955))
                    removed_elements = 0

                    # reproject detections
                    for j in range(len(detected)):
                        pan_point_1 = from_cube2panoramic(face, [detected[j][0], detected[j][1]])
                        pan_point_2 = from_cube2panoramic(face, [detected[j][2], detected[j][3]])

                        # check if it cross the edge
                        if abs(pan_point_2[0] - pan_point_1[0]) < self.width/ 2:
                            rep_detections.append([pan_point_1[0], pan_point_1[1], pan_point_2[0], pan_point_2[1]])
                        else:
                            del border_det[j - removed_elements]
                            removed_elements += 1

                    inside_detections += border_det

                # plot side
                if out_path_det is not None:
                    plot_detection(cv_image, detected, face + str(frame).zfill(4), out_path_det + "/side/" + face + "_")

        # plot only border
        outside_detections = [not d for d in inside_detections]
        rep_inside = np.array(rep_detections)[inside_detections]

        # procedure to merge bordere detections
        rep_outside = np.array(rep_detections)[outside_detections]
        merged = [False for j in range(len(rep_outside))]
        new_detections = []
        for k1 in range(len(rep_outside) - 1):
            for k2 in range(k1 + 1, len(rep_outside)):
                # check if to merge
                if not merged[k1]:
                    if 0 < (rep_outside[k2][0] - rep_outside[k1][2]) < 20:
                        # merge
                        new_detections.append(
                            [rep_outside[k1][0], min(rep_outside[k1][1], rep_outside[k2][1]), rep_outside[k2][2],
                             max(rep_outside[k1][3], rep_outside[k2][3])])
                        merged[k1] = True
                        merged[k2] = True
                        continue
                    elif 0 < (rep_outside[k1][0] - rep_outside[k2][2]) < 20:

                        new_detections.append(
                            [rep_outside[k2][0], min(rep_outside[k1][1], rep_outside[k2][1]), rep_outside[k1][2],
                             max(rep_outside[k1][3], rep_outside[k2][3])])
                        merged[k1] = True
                        merged[k2] = True
                        continue
            if not merged[k1]:
                new_detections.append(rep_outside[k1])

        if len(merged) > 0:
            if not merged[-1]:
                new_detections.append(rep_outside[-1])
        if out_path_det is not None:
            for det in rep_inside:
                rep_image = cv2.rectangle(rep_image, det[:2], det[2:], (0, 255, 0), 5)
            for det in new_detections:
                rep_image = cv2.rectangle(rep_image, det[:2], det[2:], (255, 0, 0), 5)

            cv2.imwrite(out_path_det+'/rep/rep_' + str(str(frame).zfill(4)) + '.jpg', cv2.cvtColor(rep_image, cv2.COLOR_BGR2RGB))

        if len(new_detections) > 0:
            rep_detections = np.concatenate((new_detections, rep_inside))
        else:
            rep_detections = rep_inside

        tracking_out = []

        if self.tracking:
            track_det = np.array(rep_detections)
            track_det[:, 2] = track_det[:, 2] - track_det[:, 0]
            track_det[:, 3] = track_det[:, 3] - track_det[:, 1]

            raw_det = []
            for det in track_det:
                raw_det.append((det, 1, 0))

            tracks = self.tracker.update_tracks(raw_det, frame=cv2_image_rgb)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                tracking_out.append([track_id, ltrb.astype(int)])

                if out_path_det is not None:
                    rep_image_track = cv2.rectangle(rep_image_track, ltrb[:2].astype(int), ltrb[2:].astype(int), (0, 255, 0), 5)
                    rep_image_track = cv2.putText(rep_image_track, track_id, ltrb[:2].astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if out_path_det is not None:
                cv2.imwrite(out_path_det + '/track/track_' + str(str(frame).zfill(4)) + '.jpg', cv2.cvtColor(rep_image_track, cv2.COLOR_BGR2RGB))

        return rep_detections, face_detections, tracking_out

def load_model(weights = "./src/yolov7.pt"):

    imgsz = 640
    # Initialize
    set_logging()
    device = select_device('0')
    #half = device.type != 'cpu'  # half precision only supported on CUDA
    print(imgsz)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    trace = True
    if trace:
        model = TracedModel(model, device, imgsz)

    #if half:
    #    model.half()  # to FP16
    return model


def detect_person(img0, model):
    poses = []
    obejcts_poses = []
    imgsz = 640
    conf_thres = 0.75
    iou_thres = 0.45
    device = select_device('')
    stride = int(model.stride.max())  # model stride
    half = device == 'cuda'  # Set 'half' to True if using GPU, False if using CPU
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    t3 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = img0
        s = ''
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format

                if names[int(cls)] == 'person':
                    # print(names[int(cls)])
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
                    poses.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                else:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
                    obejcts_poses.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

    # cv2.imshow("image", img0)
    # cv2.imwrite('/home/sepid/Pictures/output.jpg', img0)
    # cv2.waitKey(0)
    # print(poses)
    return poses, obejcts_poses



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

    imdet = ImageDetector()

    for i in range(30, len(ids)):
        scan = scans[i]
        id = ids[i]

        img_path = osp.join(path_images, str(id).zfill(4)+".png")
        print(img_path)

        # read panoramic
        img1 = cv2.imread(img_path)
        cv2_image_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        imdet.compute_detections_sides(cv2_image_rgb, out_path_det)

