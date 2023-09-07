import csv
import os.path as osp
import numpy as np

from sklearn.cluster import DBSCAN

from src.auto_labeling_tools.util.visualization import plot_scans
from src.auto_labeling_tools.util.laser_detector import LaserDetector


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

    path_bag = "/media/leonardo/Elements/bag/hospital1_static.bag"
    t_laser = "/scan"
    t_img = "/theta_camera/image_raw"
    path_out = "/media/leonardo/Elements/prova"
    #path_dr_spaam = "./dr_spaam_e40.pth"
    path_dr_spaam = "./ckpt_jrdb_pl_dr_spaam_phce_mixup_e20.pth"
    img_out_path = "/media/leonardo/Elements/prova/img_out"
    laser_spec = {
        'frame_id': "base_link",
        'angle_min': -3.140000104904175,
        'angle_max': 3.140000104904175,
        'angle_increment': 0.005799999926239252,
        'range_min': 0.44999998807907104,
        'range_max': 25.0,
        'len': 1083
    }

    path_laser = osp.join(path_out, "laser.csv")
    path_images = osp.join(path_out, "img")

    scans, ids = read_scan(path_laser)

    ld = LaserDetector(scans, laser_spec)

    points_wall = ld.get_points_wall()

    plot_scans(points_wall, None, img_out_path, str(-1).zfill(4))

    for i in range(len(ids)):
        id = ids[i]
        scan = scans[i]

        people, labels, points, cluster_centroids = ld.detect_people(scan)
        if len(labels) > 0:
            plot_scans(points, people, img_out_path, str(id).zfill(4), labels)

        '''
        detector = Detector(
            path_dr_spaam,
            model="DR-SPAAM",
            #ckpt_file=path_dr_spaam,
            gpu=True,
            stride=1,
            # tracking=False,
            panoramic_scan=True
        )
    
        laser_fov_deg = 360
        detector.set_laser_fov(laser_fov_deg)
            
        dets_xy, dets_cls, instance_mask = detector(scan)  # get detection
        # confidence threshold
        cls_thresh = 0.2
        cls_mask = dets_cls > cls_thresh
        dets_xy = dets_xy[cls_mask]
        dets_cls = dets_cls[cls_mask]
        '''


