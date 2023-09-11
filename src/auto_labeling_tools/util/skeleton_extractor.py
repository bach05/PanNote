import os
import numpy as np

from mmpose.apis import MMPoseInferencer

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



if __name__ == "__main__":
    img_path = '/media/leonardo/Elements/lab_outdoor_1_2/img'  # replace this with your own image path
    out_file = "/media/leonardo/Elements/lab_outdoor_1_2/out_skeleton.csv"
    path_boxes = "/media/leonardo/Elements/lab_outdoor_1_2//automatic_annotations.csv"
    # instantiate the inferencer using the model alias
    inferencer = MMPoseInferencer('human')

    file_out = open(out_file, "w")

    # The MMPoseInferencer API employs a lazy inference approach,
    # creating a prediction generator when given input
    dict = {}
    file = open(path_boxes, "r")
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

        boxes = np.array(boxes).astype(float)

        path_image = os.path.join(img_path, image+".png")
        result_generator = inferencer(img_path, show=False)
        results = next(result_generator)

        predictions = results["predictions"][0]

        asso = np.zeros((len(boxes), len(predictions)))

        for k1, prediction in enumerate(predictions):
            box_sk = prediction['bbox']
            for k2, box in enumerate(boxes):
                asso[k2, k1] = calculate_iou(box, box_sk[0])

        ann_assosiations = np.argmax(asso, axis=1)
        i, c = np.unique(ann_assosiations, return_counts=True)
        correct_boxes = c == 1

        if np.count_nonzero(correct_boxes) > 0:
            boxes = boxes.astype(str)
            file_out.write(image)

            for k3 in range(len(boxes)):
                if correct_boxes[np.where(i == ann_assosiations[k3])]:
                    box = boxes[k3]
                    centroid = centroids[k3]
                    prediction = predictions[ann_assosiations[k3]]
                    sk = np.array(prediction["keypoints"]).astype(str)
                    conf = np.array(predictions[1]["keypoint_scores"]).astype(str)

                    file_out.write(";"+box[0]+";"+box[1]+";"+box[2]+";"+box[3]+";"+centroid[0]+";"+centroid[1])
                    for k4 in range(len(sk)):
                        file_out.write(";" +sk[k4, 0]+";" + sk[k4, 1]+";" + conf[k4])

            file_out.write("\n")


        file_out.flush()
        os.fsync(file_out.fileno())


    file.close()


    '''
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
                    if abs(pan_point_2[0] - pan_point_1[0]) < self.width / 2:
                        rep_detections.append([pan_point_1[0], pan_point_1[1], pan_point_2[0], pan_point_2[1]])
                    else:
                        del border_det[j - removed_elements]
                        removed_elements += 1

                inside_detections += border_det

            # plot side
            if out_path_det is not None:
                plot_detection(cv_image, detected, face + str(frame).zfill(4), out_path_det + "/side/" + face + "_")
    
    '''