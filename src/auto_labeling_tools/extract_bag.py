
import os.path as osp
from util.bag_extractor import Bag_manager



if __name__ == "__main__":

    path_bag = "/media/leonardo/Elements/bag/hospital3_static.bag"
    t_laser = "/scan"
    t_img = "/theta_camera/image_raw"
    path_out = "/media/leonardo/Elements/prova"

    path_laser = osp.join(path_out, "laser.csv")
    path_images = osp.join(path_out, "img")

    # extract data from bag
    bm = Bag_manager(path_bag, t_img, t_laser)
    bm.extract_bag(path_out)

