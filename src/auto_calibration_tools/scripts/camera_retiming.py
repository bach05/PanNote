#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image


def scan_callback(original_image):
    # Create a new LaserScan message with a different header
    new_scan = Image()
    new_scan = original_image
    new_scan.header.stamp = rospy.Time.now() - rospy.Duration(1.011) # Use the current time as the new timestamp

    # Publish the new scan message on the \scan2 topic
    scan2_publisher.publish(new_scan)

if __name__ == '__main__':
    rospy.init_node('cam_republisher_node')

    # Define the topic names
    original_scan_topic = "/theta_camera/image_raw"
    new_scan_topic = "/theta_camera/image_raw2"

    # Subscribe to the original scan topic
    rospy.Subscriber(original_scan_topic, Image, scan_callback)

    # Create a publisher for the new scan topic
    scan2_publisher = rospy.Publisher(new_scan_topic, Image, queue_size=10)

    rospy.spin()
