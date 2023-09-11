#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import csv
import threading
import subprocess
from std_srvs.srv import SetBool

class SynchNode:

    def __init__(self):
        rospy.init_node('sync_node', anonymous=True)

        # Retrieve parameters
        SAVE_ROOT = rospy.get_param("~save_dir", "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/manual_labelling/hospital3_static")
        #num_images_to_save = rospy.get_param("~num_images_to_save", 100)
        save_interval = rospy.get_param("~save_step", 1)  # Adjust the interval as needed (in seconds)

        # Specify the service name
        service_name = '/bag_player/pause_playback'
        # Create a proxy for the service
        rospy.wait_for_service(service_name)
        self.pause_playback = rospy.ServiceProxy(service_name, SetBool)

        if not os.path.exists(SAVE_ROOT):
            os.makedirs(SAVE_ROOT)
            print(f"Folder {SAVE_ROOT} created.")
            os.makedirs(os.path.join(SAVE_ROOT,"img"))
            print(f"Folder {os.path.join(SAVE_ROOT,'img')} created.")
        else:
            print(f"Folder '{SAVE_ROOT}' already exists.")

        csv_file_path = os.path.join(SAVE_ROOT, "laser.csv")
        self.csv_file = open(csv_file_path, 'w')
        self.csvwriter = csv.writer(self.csv_file)

        self.save_step = save_interval  # Seconds
        self.save_root = SAVE_ROOT

        # Initialize variables to store data
        self.bridge = CvBridge()

        # Create subscribers for the topics
        self.image_sub = Subscriber("/theta_camera/image_raw2", Image, queue_size=30)
        self.scan_sub = Subscriber("/scan2", LaserScan, queue_size=30)

        # Synchronize the messages from both topics
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.scan_sub], queue_size=30, slop=0.04)
        self.sync.registerCallback(self.callback)

        self.image_count = 0
        self.images = []
        self.scans = []
        self.ids = []
        self.stopped_bag = False

        # Create a lock to synchronize access to self.images and self.scans
        self.data_lock = threading.Lock()

        # Create a thread to run the saver() method
        self.saver_thread = threading.Thread(target=self.saver)
        #self.saver_thread.daemon = True  # Allow the program to exit even if this thread is running
        self.saver_thread.start()  # Start the thread


    def callback(self, image_msg, scan_msg):
        # Store the received data

        image_data = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        laser_ranges = scan_msg.ranges

        ts_img = image_msg.header.stamp.to_sec()
        ts_laser = scan_msg.header.stamp.to_sec()

        if self.image_count % self.save_step == 0:
            #print(f"***RECEIVED {self.image_count}:\nL [{ts_laser}]\nC [{ts_img}]\ndelta L-C {ts_laser-ts_img} ")
            # image_filename = str(self.image_count).zfill(4)+ '.png'
            # image_path = os.path.join(self.save_root, "img", image_filename)
            # cv2.imwrite(image_path, image_data)
            # self.csvwriter.writerow([str(self.image_count).zfill(4), ','.join(map(str, laser_ranges))])
            self.images.append(image_data)
            self.scans.append(laser_ranges)
            self.ids.append(self.image_count)

        self.image_count +=1

    def saver(self):
        while True:
            # Check if there is data to save
            with self.data_lock:
                if len(self.images) > 0 and len(self.scans) > 0:
                    image_data = self.images.pop(0)
                    laser_ranges = self.scans.pop(0)
                    id = self.ids.pop(0)
                else:
                    # If there's no data, sleep for a short time to avoid busy-waiting
                    rospy.sleep(0.01)
                    continue

            if len(self.images) > 100 and ~self.stopped_bag:
                # Pause the bag playback when the queue size exceeds 100
                self.pause_playback(data=True)
                print("Bag playback paused.")
                self.stopped_bag = True

            print(f"***SAVED {id}, queue size {len(self.images)}")
            image_filename = str(id).zfill(4) + '.png'
            image_path = os.path.join(self.save_root, "img", image_filename)
            cv2.imwrite(image_path, image_data)
            self.csvwriter.writerow([str(id).zfill(4), ','.join(map(str, laser_ranges))])

            if len(self.images) < 10 and self.stopped_bag:
                # Resume the bag playback when the queue size is small enough
                self.pause_playback(data=False)
                print("Bag playback resumed.")
                self.stopped_bag = False


if __name__ == '__main__':

    try:
        node = SynchNode()
        print("Start data acquisition!")
        rospy.spin()
        node.csv_file.close()
        print(f"Analyzed {node.image_count} images")
    except rospy.ROSInterruptException:
        node.csvwriter.close()
