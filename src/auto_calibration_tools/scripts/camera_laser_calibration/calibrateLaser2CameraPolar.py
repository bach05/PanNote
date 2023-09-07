import math
import os.path

import numpy as np
import torch
from scipy.optimize import minimize, least_squares
import pickle
import cv2
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.linalg import svd
import random
from MLP import MLPTrainer
from cube_projection_LB import CubeProjection
import re
from matplotlib.collections import LineCollection

import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

from pyquaternion import Quaternion
import math
from matplotlib.cm import ScalarMappable


def undistort(img, K, D, balance=1.0, dim2=None, dim3=None):
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    DIM = dim1
    #assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1

    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img


def undistortPoints(points, K, D, dim1, balance=1.0, dim2=None, dim3=None):

    points = points.astype(np.float32)

    DIM = dim1
    #assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1

    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)

    undistorted_points = cv2.fisheye.undistortPoints(points, K, D, np.eye(3), new_K)

    return undistorted_points

from sklearn.linear_model import RANSACRegressor


SIDE_MAP = {
    'back': 0,
    'left': 1,
    'front': 2,
    'right': 3,
    'top': 4,
    'bottom': 5
}

def periodicDist(x1, x2, normalization=3840):

    x1 = x1 / normalization
    x2 = x2 / normalization

    diff = abs(x1-x2)

    if diff <= 0.5:
        dist = diff
    else:
        dist = -diff + 1

    return dist * normalization


def distErr(params_R, rho_laser, theta_laser, u_I):

    pred_R = predRANSAC(params_R, rho_laser, theta_laser)

    pred_u = 0

    error = periodicDist(pred_u, u_I)

    return error.mean()

def optimize(H0, thetaL, thetaI, error_function):

    optimization_function = lambda H: error_function(H, thetaL, thetaI)
    result = minimize(optimization_function, H0, method='Nelder-Mead')

    optimized_H = result.x
    optimized_error = result.fun

    print("Optimization res: ", result.success)
    print("Message: ", result.message)

    return optimized_H, optimized_error

def predRANSAC(ransac_params, rho_laser, theta_laser):

    a, b, c = ransac_params

    return a * rho_laser + b * theta_laser + c

if __name__ == '__main__':

    # Specify the path of the calibration data
    file_path = "cameraLaser_pointsUHD_static_indoor.pkl"

    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the dictionary from the pickle file
        data = pickle.load(file)

    outliers = [
        "image_83.png", "image_94.png", "image_96.png", "image_97.png",
        "image_98.png", "image_99.png", "image_108.png", "image_109.png",
        "image_112.png", "image_125.png", "image_126.png", "image_130.png",
        ]

    parameters = []
    image_points = []
    image_points_5 = []
    laser_points = []
    laser_point_polars = []
    names = []

    cube = CubeProjection(None, None)

    print("*****Found {} tuples".format(len(data)))

    for points in data:

        a, b, img_name, ap, bp = points

        p = [ (a,ap) ]

        for tuple in p:

            cartesian, polar = tuple
            laser_point, board_points = cartesian
            laser_point_polar, _ = polar

            u_coors = board_points[:, 0]  # Extracts the first column (u coordinates)
            v_coords = board_points[:, 1]  # Extracts the second column (v coordinates)
            X = laser_point[0]
            Y = laser_point[1]

            names.append(img_name)

            laser_points.append(np.array([X, Y, 0.0]))
            laser_point_polars.append(np.array([laser_point_polar[0], laser_point_polar[1], 0.0])) # rho, theta, phi
            image_points.append(np.array([u_coors[-1], v_coords[-1], 1])) # homogeneaus coordinates

            image_points_5.append(np.array([u_coors, v_coords, np.ones(5)])) # homogeneaus coordinates


    laser_points = (np.array(laser_points))

    laser_point_polars = (np.array(laser_point_polars))
    image_points = (np.array(image_points)).astype(np.float32)
    image_points_5 = (np.array(image_points_5)).astype(np.float32)

    #image_points[:,:2] = undistortPoints(np.expand_dims(image_points[:,:2], axis=0), K, D, (3840, 1920))

    mask = laser_point_polars[:,1] > 0

    image_points[mask, 0] = image_points[mask, 0] - 3840

    #RANSAC Linear Regression

    # Define the RANSAC model (linear regression in this case)
    ransac = RANSACRegressor()

    # Fit the RANSAC model to your 3D points
    ransac.fit(laser_point_polars[:, :2], np.expand_dims(image_points[:, 0], axis=1))

    print("RANSAC SOLUTION")
    R_params = [ransac.estimator_.coef_[0,0], ransac.estimator_.coef_[0,1], ransac.estimator_.intercept_[0]]
    print(f"{ransac.estimator_.coef_[0,0]} rho + {ransac.estimator_.coef_[0,1]} * theta + {ransac.estimator_.intercept_[0]}")

    errors = []
    pred_thetaLs = []
    rhos = []
    thetas = []

    cartesian_GT = []
    cartesian_pred = []
    image_coords = []

    preds_thetaI = []

    avg_err = 0
    cont = 0

    image_points[mask, 0] = image_points[mask, 0] + 3840

    # Get the mask of inliers
    names = np.array(names)
    inlier_mask = ransac.inlier_mask_

    wrong_detections = np.zeros_like(image_points).astype(bool)
    avg_err = []

    for i, ((rho, thetaL, _ ), (thetaI, phiI, one)) in enumerate(zip(laser_point_polars, image_points)):

        name = names[i]
        image_coord = np.array((thetaI, phiI, 1))
        #pred_thetaI = ransac.predict([[rho, thetaL]])


        pred_thetaI = predRANSAC(R_params, rho, thetaL)

        if thetaL > 0:
            pred_thetaI = pred_thetaI + 3840

        pred_thetaI = pred_thetaI % 3840

        preds_thetaI.append(pred_thetaI)

        err = periodicDist(pred_thetaI, thetaI, normalization=3840)

        name = names[i]
        im_name = os.path.basename(name)

        # if not (err < 600):
        #
        #     print(f"******** Wrong detection  in {im_name} *******+")
        #
        #     # Use regular expression to find the ID and convert it to an integer
        #     match = re.search(r'image_(\d+)\.png', im_name)
        #     wrong_detections[i] = True

        if im_name in outliers:

            print(f"******** Wrong detection  in {im_name} *******+")
            # Use regular expression to find the ID and convert it to an integer
            match = re.search(r'image_(\d+)\.png', im_name)
            wrong_detections[i] = True


        else:
            avg_err.append(err)
            cont += 1

        # #VISUAL FEEDBACK
        # fig = plt.figure(im_name, figsize=(18, 12))
        #
        # im_root = "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration/"
        # img_name = os.path.join(im_root, "imagesUHD_board_static_i", im_name)
        # pano_img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        #
        # #pano_img = undistort(pano_img, K , D)
        #
        # plt.subplot(2,1,1)
        # plt.imshow(pano_img)
        # plt.scatter(thetaI, phiI, marker="x", color="orange")
        # plt.axvline(pred_thetaI, color="red")
        #
        # plt.subplot(2,1,2)
        # im_root = "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration/pano_process/"
        # img_name = os.path.join(im_root, "indoor_static", im_name)
        # pano_img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        # plt.imshow(pano_img)
        #
        # plt.show()
        # plt.close(fig)


    # Calculate the RMSE
    preds_thetaI = np.array(preds_thetaI)
    avg_err = np.array(avg_err)

    print("AVG ABS ERR: ", avg_err.mean())

    #Plot GRAPHS
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D scatter plot

    #metric laser space
    # x = laser_points[:,0]
    # y = laser_points[:,1]

    # polar laser space
    x = laser_point_polars[:, 0]
    y = laser_point_polars[:, 1]

    z = preds_thetaI
    ax.scatter(x, y, z, c='blue', marker='x', label="pred")

    z = image_points[:,0]
    ax.scatter(x, y, z, c='red', marker='o', label="gt")

    # Set labels for the axes
    ax.set_xlabel('rho')
    ax.set_ylabel('theta')
    ax.set_zlabel('u image GT/PRED')

    plt.legend()
    plt.show()


    #################à

    # Define the lower and upper bounds for the range
    lower_bound = 0.6
    upper_bound = 0.7

    # Create a boolean mask using NumPy logical operators
    #mask = (lower_bound < np.abs(laser_point_polars[:, 1])) & (np.abs(laser_point_polars[:, 1]) < upper_bound)
    mask = np.ones_like(names).astype(bool)

    #plt.subplot(2,1,1)
    x = laser_point_polars[mask,1]
    y = image_points[mask,0]
    y2 = preds_thetaI

    # Extract the values for colormap based on laser_point_polars[:,0]
    colormap_values = laser_point_polars[mask, 0]
    plt.scatter(x,y, c=colormap_values, cmap='viridis', marker ="o", label="gt")
    plt.scatter(x,y2, c=colormap_values, cmap='viridis', marker="x", label="pred")
    plt.colorbar()

    indexes = np.where(mask)[0]

    # Loop through laser_points and names to add labels to points
    # for i in indexes:
    #     x = laser_point_polars[i,1]
    #     y = image_points[i,0]
    #     #name = "rho {:.1f},".format(laser_point_polars[i,0])
    #     name =names[i]
    #     # Define annotation properties
    #     annotation_props = {
    #         'textcoords': 'offset points',
    #         'arrowprops': {'arrowstyle': '->'},
    #         'fontsize': 8,  # Adjust the text size here
    #         'xytext': (0, 5),  # Offset for the text from the point
    #         'ha': 'center',
    #     }
    #
    #     plt.annotate(name, (x, y), **annotation_props)

    plt.xlabel("theta Laser")
    plt.ylabel("u image")

    plt.legend()
    plt.show()


    ##### ERROR PLOT

    fig = plt.figure("2D")

    ax = fig.add_subplot(211)
    x = laser_point_polars[~wrong_detections[:,0], 0]
    ax.scatter(x, avg_err, c="orange", marker='+', label="gt")
    ax.set_xlabel('rho laser')
    ax.set_ylabel('error')

    ax = fig.add_subplot(212)
    x = laser_point_polars[~wrong_detections[:,0], 1]
    ax.scatter(x, avg_err, c="orange", marker='+', label="gt")
    ax.set_xlabel('theta laser')
    ax.set_ylabel('error')

    plt.show()


    results = {
        "ransac":R_params
    }

    # Specify the file path where you want to save the dictionary
    file_path = "laser2camera_polar_map.pkl"

    # save dictionary to person_data.pkl file
    with open(file_path, 'wb') as fp:
        pickle.dump(results, fp)
        print('Dictionary saved successfully to file')

    print(f"Laser to Camera calibration results saved to {file_path}")






