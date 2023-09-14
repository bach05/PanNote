import numpy as np
import os


def calculate_iou(box1, box2):
    # Box format: (x1, y1, x2, y2)
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the coordinates of the intersection rectangle
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Check if there is an intersection (i.e., the boxes overlap)
    if x1_i < x2_i and y1_i < y2_i:
        # Calculate the area of intersection
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate the area of each bounding box
        area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate the union area
        union_area = area_box1 + area_box2 - intersection_area

        # Calculate the IoU
        iou = intersection_area / union_area
        return iou
    else:
        return 0.0  # No overlap, IoU is 0

def read_data(path):

    dic_data = {}
    file = open(path, "r")
    for line in file:
        s = line.split(";")
        image = s[0]
        data = s[1:]

        boxes = []
        centroids = []
        for i in range(int(len(data)/6)):
            person = data[i*6:(i+1)*6]
            box = person[:4]
            centroid = person[4:]

            boxes.append(box)
            centroids.append(centroid)

        dic_data[image] =[np.array(boxes).astype(int), np.array(centroids).astype(float)]
    file.close()

    return dic_data


if __name__ == "__main__":


    base_path = "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/bag_extraction/lab_indoor_3_2"
    path_gt = os.path.join(base_path, "manual_ann_"+"lab_indoor_3_2.csv")
    path_annotation = os.path.join(base_path,"out/automatic_annotations.csv")

    annotated_data = {}

    gt = read_data(path_gt)
    annotated_images = list(gt.keys())

    ann_file = open(path_annotation, "r")

    distances = []
    correct_1 = 0
    correct_075 = 0
    correct_05 = 0
    correct_025 = 0
    correct_010 = 0

    total_boxes_gt = 0
    total_boxes_annotated = 0
    matched_boxes = 0

    for line in ann_file:
        s = line.split(";")
        image = s[0]
        data = s[1:]

        if image in annotated_images:

            boxes_gt = gt[image][0]
            centroids_gt = gt[image][1]

            total_boxes_gt += len(boxes_gt)
            total_boxes_annotated += int(len(data) / 6)

            boxes = []
            centroids = []
            for i in range(int(len(data) / 6)):
                person = data[i * 6:(i + 1) * 6]
                box = person[:4]
                centroid = person[4:]

                boxes.append(box)
                centroids.append(centroid)

                np_box = np.array(box).astype(int)
                corresponding = np.where((boxes_gt[:, 0] == np_box[0]) & (boxes_gt[:, 1] == np_box[1]) & (boxes_gt[:, 2] == np_box[2]) & (boxes_gt[:, 3] == np_box[3]))
                if len(corresponding[0]) > 0:
                    index = corresponding[0][0]
                    dist = np.linalg.norm((np.array(centroid).astype(float) - centroids_gt[index]))
                    distances.append(dist)
                    if dist < 0.10:
                        correct_1 += 1
                        correct_075 += 1
                        correct_05 += 1
                        correct_025 += 1
                        correct_010 += 1
                    elif dist < 0.25:
                        correct_1 += 1
                        correct_075 += 1
                        correct_05 += 1
                        correct_025 += 1
                    elif dist < 0.5:
                        correct_1 += 1
                        correct_075 += 1
                        correct_05 += 1
                    elif dist < 0.75:
                        correct_1 += 1
                        correct_075 += 1
                    elif dist < 1:
                        correct_1 += 1

                    matched_boxes += 1

            # boxes = np.array(boxes).astype(int)
            # centroid = np.array(centroid).astype(float)

            print("Found ", image)
        else:
            print("Annotation not found for image: ", image)

    distances = np.array(distances)

    print("Mean error: ", np.mean(distances))
    print("Acc 1: ", correct_1/matched_boxes)
    print("Acc 0.75: ", correct_075/matched_boxes)
    print("Acc 0.5: ", correct_05/matched_boxes)
    print("Acc 0.25: ", correct_025/matched_boxes)
    print("Acc 0.10: ", correct_010/matched_boxes)

    print("Total gt ", total_boxes_gt, " Total annotated: ", total_boxes_annotated, " matched: ", matched_boxes)

    # Open the file for writing
    output_file_path = os.path.join(base_path, "results_annotations.txt")
    with open(output_file_path, 'w') as output_file:
        # Write the output to the file
        output_file.write(f"Mean error: {np.mean(distances)}\n")
        output_file.write(f"Acc 1: {correct_1 / matched_boxes}\n")
        output_file.write(f"Acc 0.75: {correct_075 / matched_boxes}\n")
        output_file.write(f"Acc 0.5: {correct_05 / matched_boxes}\n")
        output_file.write(f"Acc 0.25: {correct_025 / matched_boxes}\n")
        output_file.write(f"Acc 0.10: {correct_010 / matched_boxes}\n")
        output_file.write(
            f"Total gt {total_boxes_gt}, Total annotated: {total_boxes_annotated}, matched: {matched_boxes}\n")

