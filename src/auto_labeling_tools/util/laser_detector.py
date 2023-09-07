import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import MeanShift, DBSCAN

class LaserDetector:
    def __init__(self, scans, laser_spec, scale=1000, dist_t=0.5):
        self.scans = scans
        self.laser_spec = laser_spec
        self.dist_t=dist_t

        disc_mat = np.zeros((1083, scale + 1))
        for scan in scans:
            index_r = (scan / 26 * scale).astype(int)
            index_t = np.arange(1083)
            disc_mat[index_t, index_r] += 1

        self.wall = np.argmax(disc_mat, axis=1)
        self.wall = self.wall.astype(float) / scale * 26

        self.points_wall, self.polar_wall = convert_to_coordinates(self.wall, self.laser_spec)

        self.tree = KDTree( self.points_wall)

    def get_points_wall(self):
        return self.points_wall.copy()

    def detect_people(self, scan):
        points, polar = convert_to_coordinates(scan, self.laser_spec, remove_out=True)
        dd, ii = self.tree.query(points, k=1)

        people_points = points[dd > 0.5]

        # clustering = MeanShift(bandwidth=2).fit(people_points)
        if len(people_points) > 0:
            clustering = DBSCAN(eps=0.4, min_samples=1).fit(people_points)
            labels = clustering.labels_

            unique_labels = set(labels) - {-1}  # Exclude noise points
            cluster_centroids = []

            for label in unique_labels:
                cluster_points = people_points[labels == label]
                cluster_centroid = np.mean(cluster_points, axis=0)
                cluster_centroids.append(cluster_centroid)

            cluster_centroids = np.array(cluster_centroids)
        else:
            labels = []
            cluster_centroids = []


        return people_points, labels, points, cluster_centroids


def convert_to_coordinates(ranges_list, laser_spec, remove_out=True):
    ranges = np.array(ranges_list)

    # Calculate the angles for each range measurement
    angles = np.arange(laser_spec['angle_min'], laser_spec['angle_max'], laser_spec['angle_increment'])

    if remove_out:
        mask = ranges <= laser_spec['range_max']
        ranges = ranges[mask]
        angles = angles[mask]

    # Convert polar coordinates to Cartesian coordinates
    x = np.multiply(ranges, np.cos(angles))
    y = np.multiply(ranges, np.sin(angles))

    points = np.stack([x, y], axis=1)
    points_polar = np.stack([ranges, angles], axis=1)

    return points, points_polar
