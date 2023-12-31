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

from matplotlib.collections import LineCollection

import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

from pyquaternion import Quaternion
import math


def undistort(img, K, D, balance=0.0, dim2=None, dim3=None):
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


def undistortPoints(points, K, D, dim1, balance=0.0, dim2=None, dim3=None):

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


def rmse(objp, imgp, lens, K, D, rvec, tvec):
    if lens == 'pinhole':
        predicted, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        predicted = cv2.undistortPoints(predicted, K, D, P=K)
    elif lens == 'fisheye':
        predicted, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, K, D)
        predicted = cv2.fisheye.undistortPoints(predicted, K, D, P=K)
    predicted = predicted.squeeze()
    imgp = imgp.squeeze()

    pix_serr = []
    for i in range(len(predicted)):
        xp = predicted[i,0]
        yp = predicted[i,1]
        xo = imgp[i,0]
        yo = imgp[i,1]
        pix_serr.append((xp-xo)**2 + (yp-yo)**2)
    ssum = sum(pix_serr)

    return math.sqrt(ssum/len(pix_serr))

SIDE_MAP = {
    'back': 0,
    'left': 1,
    'front': 2,
    'right': 3,
    'top': 4,
    'bottom': 5
}
#
# def project(H, xt, yt):
#
#     a0, b0, c0, a1, b1, c1 = H
#
#     u = a0 * xt + b0 * yt + c0
#     v = a1*xt + b1 * yt + c1
#
#     return u,v
#
#
# def projectFull(H, X, Y):
#
#     w11, w12, w21, w22, w31, w32, t1, t2, t3 = H
#
#     u = (fu * w11 + u0 * w31) * X + (fu * w12 + u0 * w32) * Y + fu * t1 + u0 * t3
#     v = (fv * w21 + v0 * w31) * X + (fv * w22 + v0 * w32) * Y + fv * t2 + v0 * t3
#
#     return u,v
#
#
# def projectVLINE(H, X, Y):
#     # w11, w12, w31, w32, t1, t3 = H
#
#     w11, w12, w31, w32, t1, t3 = H
#
#     u = (fu * w11 + u0 * w31) * X + (fu * w12 + u0 * w32) * Y + fu * t1 + u0 * t3
#
#     return u

def correctU(H, pred_u, x_3d, y_3d):

    theta, scale = H

    t = rotated_arctan2(x_3d, y_3d, theta) / (np.pi)

    correction_u = pred_u + t*scale

    return correction_u

def correctU2(H, pred_u, x_3d, y_3d):

    theta, scale, a = H

    t = rotated_arctan2(x_3d, y_3d, theta) / (np.pi)


    correction_u = pred_u + t*scale + a*(x_3d**2 + y_3d**2)

    return correction_u

def distErr(H, gt_u, pred_u, x_3d, y_3d):

    correct_u = correctU(H, pred_u, x_3d, y_3d)

    error = periodicDist(correct_u, gt_u)

    return error.mean()

def optimize(H0, gt_u, pred_u, x_3d, y_3d, error_function = distErr):

    optimization_function = lambda H: error_function(H, gt_u, pred_u, x_3d, y_3d)
    result = minimize(optimization_function, H0, method='Nelder-Mead')

    optimized_H = result.x
    optimized_error = result.fun

    print("Optimization res: ", result.success)
    print("Message: ", result.message)

    return optimized_H, optimized_error

def rotated_arctan2(x1, x2, theta):
    # Calculate the arctan2 angle
    angle = np.arctan2(x1, x2)

    # Add the rotation offset (theta) to the angle
    rotated_angle = angle + theta

    # Ensure the angle is within the range [-pi, pi]
    rotated_angle = (rotated_angle + np.pi) % (2 * np.pi) - np.pi

    return rotated_angle

# def cartesian_to_polar(cartesian_array):
#     # Ensure the input array has the shape Nx3
#     if cartesian_array.shape[-1] != 3:
#         raise ValueError("Input array must have shape Nx3")
#
#     # Extract x, y, and z components
#     x, y, z = cartesian_array[:, 0], cartesian_array[:, 1], cartesian_array[:, 2]
#
#     # Calculate rho (distance from origin)
#     rho = np.sqrt(x**2 + y**2 + z**2)
#
#     # Calculate phi (polar angle from positive z-axis, range [0, pi])
#     phi = np.arccos(z / rho)
#
#     # Calculate theta (azimuthal angle in the xy-plane, range [0, 2*pi])
#     theta = np.arctan2(y, x)
#
#     return np.column_stack((theta, phi, rho))

def periodicDist(x1, x2, normalization=3840):

    x1 = x1 / normalization
    x2 = x2 / normalization

    diff = np.absolute(x1-x2)

    mask = diff > 0.5

    diff[mask] = -diff[mask] + 1

    return diff * normalization


def is_function(x, y, z):
    # Check if the shapes of x, y, and z match
    if x.shape != y.shape or x.shape != z.shape:
        return False

    # Combine x, y, and z into a single array for easy processing
    xyz = np.column_stack((x, y, z))

    # Check for duplicate (x, y) pairs
    unique_xy = np.unique(xyz[:, :2], axis=0)

    # If the number of unique (x, y) pairs is equal to the number of data points,
    # then z = f(x, y) is a function
    return unique_xy.shape[0] == xyz.shape[0]


if __name__ == '__main__':

    ####### CAMERA MATRIX (TRIAL)
    calib_file = "../calibration_data_intrinsics/intrinsicsUHD_ball_indoor2_fisheye.pkl"
    # Open the pickle file for reading in binary mode
    with open(calib_file, 'rb') as file:
        # Load the dictionary from the pickle file
        camera_calib = pickle.load(file)

    # Specify the path of the calibration data
    file_path = "cameraLaser_pointsUHD_pano_outdoor.pkl"

    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the dictionary from the pickle file
        data = pickle.load(file)

    points_2d = []
    points_3d = []
    names = []

    cube = CubeProjection(None, None)

    print("*****Found {} tuples".format(len(data)))

    for points in data:

        a, b, img_name, ap, bp = points

        p = [(a, ap)]

        for tuple in p:
            cartesian, polar = tuple
            laser_point, board_points = cartesian
            laser_point_polar, _ = polar

            u_coors = board_points[:, 0]  # Extracts the first column (u coordinates)
            v_coords = board_points[:, 1]  # Extracts the second column (v coordinates)
            X = laser_point[0]
            Y = laser_point[1]

            # if laser_point_polar[0] > 2:
            #     continue
            #
            # if img_name in ["image_90.png", "image_22.png", "image_79.png", "image_60.png", "image_91.png"]:
            #     print(f"******** {img_name} ********" )
            #     print("laser: ", laser_point_polar)
            #     print("image: ", board_points)

            names.append(img_name)

            points_3d.append(np.array([X, Y, 0.2]))
            #laser_point_polars.append(np.array([laser_point_polar[0], laser_point_polar[1], 0.0]))  # rho, theta, phi
            points_2d.append(np.array([u_coors[-1], v_coords[-1]]))  # homogeneaus coordinates



    # laser_points = np.expand_dims(np.array(laser_points).astype(np.float32), axis=0)
    # image_points = np.expand_dims(np.array(image_points).astype(np.float32), axis=0)

    points_2d = np.expand_dims(np.array(points_2d).astype(np.float32), axis=0)
    points_3d = np.expand_dims(np.array(points_3d).astype(np.float32), axis=0)

    #points_3d[0] = cartesian_to_polar(points_3d[0])

    #mask = points_2d[0, :, 0] < 3840/2
    #points_2d[0, mask, 0] = points_2d[0, mask, 0] + 3840

    K = camera_calib['K'].astype(np.float32)
    D = camera_calib['dist'].astype(np.float32)
    D = D[:,:4]


    # K =np.array([ [2243.5442504882812, 0.0, 1945.6583862304688], [0.0, 2243.52978515625, 1022.6772766113281], [0.0, 0.0, 1.0]]).astype(np.float32)
    # D = np.array([0.0761934369802475, -0.10407509654760361, 0.042357057332992554, 0.0, 0.0]).astype(np.float32)
    # D = D[:4]

    success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, jac = cv2.Rodrigues(rvec)
    q = Quaternion(matrix=rmat)
    print("Transform from camera to laser")
    print("T = ")
    print(tvec)
    print("R = ")
    print(rmat)
    print("Quaternion = ")
    print(q)

    #print("RMSE in pixel = %f" % rmse(laser_points, image_points, 'pinhole', K, D, rvec.astype(np.float32), tvec.astype(np.float32)))


    predicted, _ = cv2.projectPoints(points_3d[:,:], rvec, tvec, K, D)
    predicted = cv2.undistortPoints(predicted, K, D, P=K)
    #predicted = undistortPoints(predicted, K, D, balance=1.0, dim1=(1920, 3840))

    H0 = (np.pi/2, 3840/2)

    predicted_H0 = correctU(H0, pred_u=predicted[:, 0, 0], x_3d=points_3d[0,:,0], y_3d=points_3d[0,:,1])

    mask = predicted_H0 > 3839
    predicted_H0[mask] = predicted_H0[mask] % 3839
    mask = predicted_H0 < 0
    predicted_H0[mask] = predicted_H0[mask] + 3840

    H0g = (np.pi / 2, 3840 / 2)


    H1, _ = optimize(H0g, points_2d[0,:,0], predicted[:, 0, 0], points_3d[0,:,0], points_3d[0,:,1])

    predicted_H1 = correctU(H1, pred_u=predicted[:, 0, 0], x_3d=points_3d[0,:,0], y_3d=points_3d[0,:,1])

    mask = predicted_H1 > 3839
    predicted_H1[mask] = predicted_H1[mask] % 3839
    mask = predicted_H1 < 0
    predicted_H1[mask] = predicted_H1[mask] + 3840


    for i, (laser_p, image_p) in enumerate(zip(points_3d[0], points_2d[0])):

        # Create the 3D scatter plot
        pred_u = predicted_H1[i]

        # name = names[i]
        # im_name = os.path.basename(name)
        # im_root = "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration/"
        # img_name = os.path.join(im_root, "images_UHD_indoor", im_name)
        # pano_img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        #
        # plt.imshow(pano_img)
        # plt.axvline(pred_u, color="orange")
        # plt.scatter(image_p[0], image_p[1], marker="x", color="cyan")
        # plt.show()

        # if laser_p[1] > 0:
        #     predicted[i, 0] = predicted[i, 0] + 3840

        #predicted[i, 0] = predicted[i, 0] % 3840

        # predicted[i, 0, 0] = corr_proj
        #
        # error = cv2.norm(image_p, (predicted[i, 0]), cv2.NORM_L2)
        # reproj_errors.append(error)
        #
        # error = periodicDist(image_p[0] , predicted[i, 0, 0], 3840)
        # reproj_errors_x.append((error))
        #
        # # error = periodicDist(image_p[0],  corr_proj, 3840 )
        # # reproj_errors_correction.append(error)
        #
        # error = (image_p[1] - predicted[i, 0, 1]) ** 2
        # reproj_errors_y.append((error))

    #print(f"REPROJECTION ERROR: {reproj_errors.mean()}")
    error = distErr(H0, points_2d[0,:,0], predicted[:, 0, 0], points_3d[0,:,0], points_3d[0,:,1])
    print(f"REPROJECTION ERROR X H0 =  {H0}: {(error.mean())}")

    error = distErr(H1, points_2d[0, :, 0], predicted[:, 0, 0], points_3d[0, :, 0], points_3d[0, :, 1])
    print(f"REPROJECTION ERROR X H1 =  {H1}: {(error.mean())}")
    #print(f"REPROJECTION ERROR X correction: {(reproj_errors_correction.mean())}")
    #print(f"REPROJECTION ERROR Y: {np.sqrt(reproj_errors_y.mean())}")

    error = periodicDist(points_2d[0, :, 1], predicted[:, 0, 1], 1920)
    print(f"REPROJECTION ERROR Y: {(error.mean())}")

    rho = np.linalg.norm(points_3d[0, :, :2], axis=1)
    theta = np.arctan2(points_3d[0, :, 1], points_3d[0, :, 0])

    # Create a 3D figure
    fig = plt.figure("3D")
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D scatter plot
    x = points_3d[0, :, 0]
    y = points_3d[0, :, 1]

    z = predicted_H0
    ax.scatter(x, y, z, c="cyan", marker='x', label="pred H0")


    z = predicted_H1
    ax.scatter(x, y, z, c="blue", marker='x', label="pred H1")

    z = points_2d[0, :, 0]
    ax.scatter(x, y, z, c="orange", marker='o', label="gt")

    # Set labels for the axes
    ax.set_xlabel('x laser')
    ax.set_ylabel('y laser')
    ax.set_zlabel('u image GT/PRED')

    plt.legend()

    fig = plt.figure("2D")

    correct_u = correctU(H1,predicted[:, 0, 0], points_3d[0, :, 0], points_3d[0, :, 1])
    error = periodicDist(correct_u, points_2d[0, :, 0])
    ax = fig.add_subplot(211)
    z = points_2d[0, :, 0]
    ax.scatter(x, z, c="orange", marker='+', label="gt")
    ax.set_xlabel('rho laser')
    ax.set_ylabel('error')

    ax = fig.add_subplot(212)
    z = points_2d[0, :, 0]
    ax.scatter(y, z, c="orange", marker='+', label="gt")
    ax.set_xlabel('theta laser')
    ax.set_ylabel('error')

    result = is_function(x,y,z)
    print("Is z = f(x, y) a function?", result)

    plt.show()



