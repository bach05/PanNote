import os
import os.path as osp
import cv2
import rosbag
import csv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np


class Bag_manager:
    def __init__(self, path, img_topic, laser_topic):
        self.path = path
        self.img_topic = img_topic
        self.laser_topic = laser_topic

    def analyze_bag(self, out_path, step_size = 7):
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
                laser_time = 0

                # Initialize variable to hold image data
                # Create iterators for both topics
                laser_iterator = bag.read_messages(topics=[self.laser_topic])
                image_iterator = bag.read_messages(topics=[self.img_topic])

                laser_msg = next(laser_iterator)
                time_start = laser_msg.message.header.stamp.to_sec()

                laser_times = []
                laser_mask = []
                img_times = []

                # # Skip the first second of laser data
                cont = 0
                # for msg in laser_iterator:
                #     cont += 1
                #     laser_times.append(msg.message.header.stamp.to_sec())
                #     laser_mask.append(True)
                #     if msg.message.header.stamp.to_sec()-time_start >= 1.675:
                #         break

                print(f"Removed {cont} messages")

                plt.figure("Time")

                try:
                    for i, (laser_msg, image_msg) in tqdm(enumerate(zip(laser_iterator, image_iterator))):


                        laser_time = laser_msg.message.header.stamp.to_sec()
                        image_time = image_msg.message.header.stamp.to_sec()

                        print(f"\nREAD {i}: \nLaser {laser_time}\nImage {image_time}\nDiff {laser_time - image_time}", )


                        # # # Synchronize the messages based on their timestamps
                        while abs(laser_time - image_time) > 0.04:  # Adjust the threshold as needed
                            if laser_time < image_time:
                                laser_msg = next(laser_iterator)
                                laser_time = laser_msg.message.header.stamp.to_sec()
                                laser_times.append(laser_time)
                                laser_mask.append(True)
                                print(f"Laser >>>>>>>>")
                            else:
                                image_msg = next(image_iterator)
                                image_time = image_msg.message.header.stamp.to_sec()
                                img_times.append(image_time)
                                print(f"Image >>>>>>>>")


                        laser_times.append(laser_time)
                        img_times.append(image_time)
                        laser_mask.append(False)

                        # At this point, laser_msg and image_msg have matching timestamps
                        laser_ranges = laser_msg.message.ranges
                        image_data = bridge.imgmsg_to_cv2(image_msg.message, desired_encoding='passthrough')

                        # VISUALIZATION ##############################

                        img_times_np = np.array(img_times) - time_start
                        laser_times_np = np.array(laser_times) - time_start
                        laser_mask_np = np.array(laser_mask)

                        # plt.scatter(img_times_np, np.ones_like(img_times_np), color="red", label="image")
                        # plt.scatter(laser_times_np, np.ones_like(laser_times_np) + 1, color="blue", label="laser")
                        # plt.scatter(laser_times_np[laser_mask_np], np.ones_like(laser_times_np[laser_mask_np]) + 1, color="cyan",
                        #             label="laser")
                        #
                        # for i, j in zip(img_times_np, laser_times_np[~laser_mask_np]):
                        #     plt.plot([i, j], [1, 2], "--y")
                        #
                        # plt.show()

                        # # Create a 1x2 grid of subplots
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                        # Plot the image in the first subplot
                        ax1.imshow(image_data)
                        ax1.set_title(f"Image {im} [{image_time}]")
                        ax1.axis('off')  # Turn off axis labels and ticks

                        # Convert the list of values to a NumPy array
                        ranges = np.array(laser_ranges)

                        # Define laser specification
                        laser_spec = {
                            'frame_id': "base_link",
                            'angle_min': -3.140000104904175,
                            'angle_max': 3.140000104904175,
                            'angle_increment': 0.005799999926239252,
                            'range_min': 0.44999998807907104,
                            'range_max': 25.0
                        }

                        mask = ranges <= laser_spec['range_max']

                        # Calculate the angles for each range measurement
                        angles = np.arange(laser_spec['angle_min'], laser_spec['angle_max'], laser_spec['angle_increment'])

                        ranges = ranges[mask]
                        angles = angles[mask]

                        # Convert polar coordinates to Cartesian coordinates
                        x = np.multiply(ranges, np.cos(angles))
                        y = np.multiply(ranges, np.sin(angles))

                        points = np.stack([x, y], axis=1)

                        # Plot the laser scan data in the second subplot
                        ax2.scatter(points[:, 0], points[:, 1], marker="o", color="orange", s=4)
                        ax2.set_title(f"Laser [{laser_time}]")
                        ax2.axis('equal')
                        # Plot horizontal x-axis (red line)
                        ax2.grid()

                        print(f"COUPLED {i} ++++++")

                        # Show the plot
                        plt.tight_layout()  # Ensure proper spacing between subplots
                        # plt.legend()

                        image_filename = str(i).zfill(4) + '.png'
                        image_path = os.path.join(out_path, "img", image_filename)
                        plt.savefig(image_path)
                        plt.close(fig)

                        # SAVING ##############################

                        # Write data to CSV row

                        # if im % step_size == 0:
                        #     print(
                        #         f"[SAVING STEP {im}] \nLaser {laser_msg.message.header.stamp.to_sec()}\nImage {image_msg.message.header.stamp.to_sec()} \n*****************")
                        #
                        #     image_filename = str(im).zfill(4)+ '.png'
                        #     image_path = os.path.join(out_path, "img", image_filename)
                        #     cv2.imwrite(image_path, image_data)
                        #     csvwriter.writerow([str(im).zfill(4), ','.join(map(str, laser_ranges))])
                        #
                        # im += 1

                except StopIteration:
                    # Handle the case where one of the iterators reaches the end of the bag file
                    print("One of the iterators reached the end of the bag file")



                img_times = np.array(img_times) - time_start
                laser_times = np.array(laser_times) - time_start
                laser_mask = np.array(laser_mask)

                plt.scatter(img_times, np.ones_like(img_times), color="red", label="image")
                plt.scatter(laser_times, np.ones_like(laser_times)+1, color="blue", label="laser")
                plt.scatter(laser_times[laser_mask], np.ones_like(laser_times[laser_mask])+1, color="cyan", label="laser")

                for i,j in zip(img_times, laser_times[~laser_mask]):
                    plt.plot([i,j], [1,2], "--y")

                plt.legend()
                plt.show()

                # # Iterate through messages in the bag
                # for topic, msg, t in tqdm(bag.read_messages(topics=[self.laser_topic, self.img_topic])):
                #
                #     image_data = None
                #     #print(t)
                #     if topic == '/scan':
                #         # Assuming msg.ranges contains the laser range data
                #         laser_ranges = msg.ranges
                #         laser_time = msg.header.stamp.to_sec()
                #         #print(f"laser head:\n{msg.header}\n *****************")
                #
                #         # Only consider the laser scan data from second 4
                #         #if t.to_sec() >= 4 and t.to_sec() < 8:
                #         #   laser_ranges = msg.ranges
                #
                #     if topic == '/theta_camera/image_raw':
                #         # Convert the ROS Image message to OpenCV image
                #         image_data = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                #         image_time = msg.header.stamp.to_sec()
                #         #print(f"image head:\n{msg.header}\n *****************")
                #
                #
                #
                #     # Save the image if available
                #     if image_data is not None:
                #         # Write data to CSV row
                #         print(f"TIME SHIFT: \nLaser {laser_time}\nImage {image_time}\nDiff {laser_time-image_time}", )
                #
                #         if im % step_size == 0:
                #             image_filename = str(im).zfill(4)+ '.png'
                #             image_path = os.path.join(out_path, "img", image_filename)
                #             cv2.imwrite(image_path, image_data)
                #             csvwriter.writerow([str(im).zfill(4), ','.join(map(str, laser_ranges))])
                #
                #         im += 1

def extract_from_bag(img_topic, laser_topic, path):
    return 1