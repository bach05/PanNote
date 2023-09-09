#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import csv

class SynchNode:

    def __init__(self):
        rospy.init_node('sync_node', anonymous=True)

        # Retrieve parameters
        SAVE_ROOT = rospy.get_param("~save_dir", "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/manual_labelling/hospital3_static")
        #num_images_to_save = rospy.get_param("~num_images_to_save", 100)
        save_interval = rospy.get_param("~save_step", 15)  # Adjust the interval as needed (in seconds)

        if not os.path.exists(SAVE_ROOT):
            os.makedirs(SAVE_ROOT)
            print(f"Folder {SAVE_ROOT} created.")
            os.makedirs(os.path.join(SAVE_ROOT,"img"))
            print(f"Folder {os.path.join(SAVE_ROOT,'img')} created.")
        else:
            print(f"Folder '{SAVE_ROOT}' already exists.")

        csv_file_path = os.path.join(SAVE_ROOT, "laser.csv")
        csv_file = open(csv_file_path, 'w')
        self.csvwriter = csv.writer(csv_file)

        self.save_step = save_interval  # Seconds
        self.save_root = SAVE_ROOT

        # Initialize variables to store data
        self.bridge = CvBridge()

        # Create subscribers for the topics
        self.image_sub = Subscriber("/theta_camera/image_raw2", Image)
        self.scan_sub = Subscriber("/scan2", LaserScan)

        # Synchronize the messages from both topics
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.scan_sub], queue_size=30, slop=0.04)
        self.sync.registerCallback(self.callback)

        self.image_count = 0


    def callback(self, image_msg, scan_msg):
        # Store the received data

        image_data = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        laser_ranges = scan_msg.ranges

        ts_img = image_msg.header.stamp.to_sec()
        ts_laser = scan_msg.header.stamp.to_sec()

        if self.image_count % self.save_step == 0:
            print(f"***SAVED {self.image_count}:\nL [{ts_laser}]\nC [{ts_img}]\ndelta L-C {ts_laser-ts_img} ")
            image_filename = str(self.image_count).zfill(4)+ '.png'
            image_path = os.path.join(self.save_root, "img", image_filename)
            cv2.imwrite(image_path, image_data)
            self.csvwriter.writerow([str(self.image_count).zfill(4), ','.join(map(str, laser_ranges))])

        self.image_count +=1



if __name__ == '__main__':

    try:
        node = SynchNode()
        print("Start data acquisition!")
        rospy.spin()
        node.csvwriter.close()
    except rospy.ROSInterruptException:
        node.csvwriter.close()
