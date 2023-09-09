
import os.path as osp
from util.bag_extractor import Bag_manager



if __name__ == "__main__":

    #path_bag = "/media/iaslab/Elements/bag/hospital3_static.bag"
    path_bag = "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/bags/hospital3_static_2.bag"
    t_laser = "/scan2"
    t_img = "/theta_camera/image_raw2"
    path_out = "../manual_labelling/hospital3_static"

    path_laser = osp.join(path_out, "laser.csv")
    path_images = osp.join(path_out, "img")

    # extract data from bag
    bm = Bag_manager(path_bag, t_img, t_laser)
    #bm.extract_bag(path_out, step_size=5)
    bm.analyze_bag(path_out, step_size=5)

