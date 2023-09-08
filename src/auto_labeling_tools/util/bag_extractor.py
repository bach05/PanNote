import os
import os.path as osp
import cv2
import rosbag
import csv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tqdm import tqdm


class Bag_manager:
    def __init__(self, path, img_topic, laser_topic):
        self.path = path
        self.img_topic = img_topic
        self.laser_topic = laser_topic

    def extract_bag(self, out_path, step_size = 7):
        bridge = CvBridge()

        if not os.path.exists(out_path):
            # If it doesn't exist, create it
            os.makedirs(out_path)
            os.makedirs(os.path.join(out_path,"img"))
            print(f"Folder '{out_path}' created successfully.")
        else:
            print(f"Folder '{out_path}' already exists.")

        csvfile = osp.join(out_path, "laser.csv")
        with open(csvfile, mode='w', newline='') as csvfile:

            csvwriter = csv.writer(csvfile)
            im = 0
            with rosbag.Bag(self.path, 'r') as bag:
                # Initialize variables to hold laser range data
                laser_ranges = []

                # Initialize variable to hold image data


                # Iterate through messages in the bag
                for topic, msg, t in tqdm(bag.read_messages(topics=[self.laser_topic, self.img_topic])):
                    image_data = None
                    #print(t)
                    if topic == '/scan':
                        # Assuming msg.ranges contains the laser range data
                        laser_ranges = msg.ranges

                        # Only consider the laser scan data from second 4
                        #if t.to_sec() >= 4 and t.to_sec() < 8:
                        #   laser_ranges = msg.ranges

                    if topic == '/theta_camera/image_raw':
                        # Convert the ROS Image message to OpenCV image
                        image_data = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


                    # Save the image if available
                    if image_data is not None:
                        # Write data to CSV row
                        if im % step_size == 0:
                            image_filename = str(im).zfill(4)+ '.png'
                            image_path = os.path.join(out_path, "img", image_filename)
                            cv2.imwrite(image_path, image_data)
                            csvwriter.writerow([str(im).zfill(4), ','.join(map(str, laser_ranges))])

                        im += 1

def extract_from_bag(img_topic, laser_topic, path):
    return 1