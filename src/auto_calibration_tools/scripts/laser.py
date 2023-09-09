#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan


def scan_callback(original_scan):
    # Create a new LaserScan message with a different header
    new_scan = LaserScan()
    new_scan = original_scan
    new_scan.header.stamp = rospy.Time.now()  # Use the current time as the new timestamp

    # Publish the new scan message on the \scan2 topic
    scan2_publisher.publish(new_scan)

if __name__ == '__main__':
    rospy.init_node('scan_republisher_node')

    # Define the topic names
    original_scan_topic = "/scan"
    new_scan_topic = "/scan2"

    # Subscribe to the original scan topic
    rospy.Subscriber(original_scan_topic, LaserScan, scan_callback)

    # Create a publisher for the new scan topic
    scan2_publisher = rospy.Publisher(new_scan_topic, LaserScan, queue_size=10)

    rospy.spin()
