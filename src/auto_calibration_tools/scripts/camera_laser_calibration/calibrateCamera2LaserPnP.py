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

def correctU2(H, x_3d, y_3d):

    theta, scale, offset = H
    t = rotated_arctan2(x_3d, y_3d, theta) / (np.pi) + offset
    correction_u = t*scale

    #correction_u = rotated_arctan2(x_3d, y_3d, np.pi/2) / (np.pi) + 1.0

    return correction_u

def predictU3(H, x_3d, y_3d):

    theta, scale, offset, tx, ty = H
    t = 0.5*(rotated_arctan2(x_3d+tx, y_3d+ty, theta) / (np.pi) + 1)
    correction_u = scale *(t + offset)

    #correction_u = rotated_arctan2(x_3d, y_3d, np.pi/2) / (np.pi) + 1.0

    return correction_u

def distErr(H, gt_u, pred_u, x_3d, y_3d, pred_type = 2):

    if pred_u is not None:
        correct_u = correctU(H, pred_u, x_3d, y_3d)
    elif pred_type == 2:
        theta, scale, offset = H
        t = rotated_arctan2(x_3d, y_3d, theta) / (np.pi) + offset
        correct_u = t * scale
    elif pred_type == 3:
        correct_u = predictU3(H, x_3d, y_3d)

    error = periodicDist(correct_u, gt_u)

    return error.mean()

def optimize(H0, gt_u, pred_u, x_3d, y_3d, error_function = distErr, pred_type=2):

    optimization_function = lambda H: error_function(H, gt_u, pred_u, x_3d, y_3d, pred_type)
    result = minimize(optimization_function, H0, method='Nelder-Mead')

    optimized_H = result.x
    optimized_error = result.fun

    #print("Optimization res: ", result.success)
    #print("Message: ", result.message)

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

# def projectPoint2Image(laser_point, K, D, T, R, H, z_value):
#
#     z = np.expand_dims(np.zeros_like(laser_point[:,:,0]) + z_value, axis=-1)
#     points_3d = np.concatenate([laser_point, z], axis=-1)
#
#     predicted, _ = cv2.projectPoints(points_3d[:, :], rvec, tvec, K, D)
#     predicted = cv2.undistortPoints(predicted, K, D, P=K)
#
#     predicted[:, 0, 0] = correctU(H1, pred_u=predicted[:, 0, 0], x_3d=points_3d[0, :, 0], y_3d=points_3d[0, :, 1])
#
#     mask = predicted[:, 0, 0] > 3839
#     predicted[mask,0,0] = predicted[mask, 0, 0] % 3839
#     mask = predicted[:, 0, 0] < 0
#     predicted[mask, 0,0] = predicted[mask, 0, 0] + 3840
#
#     return predicted


def projectPoint2Image(laser_point, H):

    x_3d, y_3d = laser_point[:,0], laser_point[:,1]
    predicted = correctU2(H, x_3d, y_3d)

    mask = predicted > 3839
    predicted[mask] = predicted[mask] % 3839
    mask = predicted < 0
    predicted[mask] = predicted[mask] + 3840

    return predicted

def projectImage2Point(image_point, H):

    theta, scale, offset = H

    x_laser = np.sin((image_point/scale - offset)*np.pi)
    y_laser = np.cos((image_point/scale - offset)*np.pi)

    return np.vstack([x_laser, y_laser]).T


if __name__ == '__main__':

    ####### CAMERA MATRIX (TRIAL)
    calib_file = "../calibration_data_intrinsics/intrinsicsUHD_ball_indoor2_fisheye.pkl"
    # Open the pickle file for reading in binary mode
    with open(calib_file, 'rb') as file:
        # Load the dictionary from the pickle file
        camera_calib = pickle.load(file)

    # Specify the path of the calibration data
    file_path = "cameraLaser_pointsUHD_ball_static_i.pkl"

    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the dictionary from the pickle file
        data = pickle.load(file)

    points_2d = []
    points_3d = []
    names = []

    print("*****Found {} tuples".format(len(data)))

    for points in data:

        img_name, point2d, point3d = points

        points_2d.append(point2d)
        points_3d.append(point3d)
        names.append(img_name)


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

    #use argtan 2 directly
    H0g = (np.pi / 2, 3840 / 2, 1.0)

    H2, _ = optimize(H0g, points_2d[0, :, 0], None, points_3d[0, :, 0], points_3d[0, :, 1])

    pred_H2 = correctU2(H2, points_3d[0, :, 0], points_3d[0, :, 1])

    mask = pred_H2 > 3839
    pred_H2[mask] = pred_H2[mask] % 3839
    mask = pred_H2 < 0
    pred_H2[mask] = pred_H2[mask] + 3840

    #USE H3 with initialization

    S = 3840
    atans = np.arctan2(points_3d[0, :, 0], points_3d[0, :, 1])

    best_error = np.inf
    best_H3 = [0,0,0,0,0]

    for alpha in [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5]:
        for theta in np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])*np.pi:

            H0g = (theta, S, alpha, 0 , 0)
            H3, _ = optimize(H0g, points_2d[0, :, 0], None, points_3d[0, :, 0], points_3d[0, :, 1], pred_type=3)

            pred_H3 = predictU3(H3, points_3d[0, :, 0], points_3d[0, :, 1])

            mask = pred_H3 > 3839
            pred_H3[mask] = pred_H3[mask] % 3839
            mask = pred_H3 < 0
            pred_H3[mask] = pred_H3[mask] + 3840

            error = periodicDist(pred_H3, points_2d[0, :, 0]).mean()

            if error < best_error and error >= 0:
                best_error = error
                best_H3 = H3


    H3 = best_H3


    for i, (laser_p, image_p) in enumerate(zip(points_3d[0], points_2d[0])):

        # Create the 3D scatter plot
        pred_u = predicted_H1[i]

        # name = names[i]
        # im_name = os.path.basename(name)
        # im_root = "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration/"
        # img_name = os.path.join(im_root, "imagesUHD_ball_400i", im_name)
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

    laser_points = points_3d[:, :, :2]
    z_value = np.mean(points_3d[:, :, 2])
    pred = projectPoint2Image(laser_point=laser_points[0],H=H2)
    error = periodicDist(pred,  points_2d[0,:,0])
    print(f"REPROJECTION ERROR X H2 =  {H2}: {(error.mean())}")

    pred = predictU3(H3, points_3d[0, :, 0], points_3d[0, :, 1])
    error = periodicDist(pred, points_2d[0, :, 0])
    print(f"REPROJECTION ERROR X H3 =  {H3}: {(error.mean())}")

    reprojected_laser = projectImage2Point(pred, H2)
    error = np.linalg.norm(reprojected_laser - points_3d[0,:,:2], axis=1).mean()
    print(f"REPROJECTION ERROR LASER: {(error.mean())}")

    rho = np.linalg.norm(points_3d[0, :, :2], axis=1)
    theta = np.arctan2(points_3d[0, :, 1], points_3d[0, :, 0])

    # Create a 3D figure
    fig = plt.figure("3D")
    ax = fig.add_subplot(111, projection='3d')

    # Create the 3D scatter plot
    x = points_3d[0, :, 0]
    y = points_3d[0, :, 1]

    #z = predicted_H0
    # z = predicted[:, 0, 0]
    # ax.scatter(x, y, z, c="red", marker='x', label="pred")
    # #ax.plot_trisurf(x, y, z, color="red", alpha=0.2)
    #
    # z = predicted_H0
    # ax.plot(x, y, z, c="cyan", marker='x', label="H0")
    #
    # laser_points = points_3d[:,:,:2]
    # z_value = np.mean(points_3d[:,:,2])
    # z = projectPoint2Image(laser_point=laser_points, K=K, D=D, T=tvec, R=rvec, H=H1, z_value=z_value)[:,0,0]
    # ax.scatter(x, y, z, c="blue", marker='x', label="pred H1")

    laser_points = points_3d[:, :, :2]
    z_value = np.mean(points_3d[:, :, 2])
    z = pred_H2
    ax.scatter(x, y, z, c="purple", marker='+', label="pred H2")

    z = points_2d[0, :, 0]
    ax.scatter(x, y, z, c="orange", marker='o', label="gt")
    #ax.plot_trisurf(x, y, z, color="orange", alpha=0.2)

    # Set labels for the axes
    ax.set_xlabel('x laser')
    ax.set_ylabel('y laser')
    ax.set_zlabel('u image GT/PRED')

    plt.legend()

    fig = plt.figure("ERROR IMAGE")

    pred = projectPoint2Image(laser_point=laser_points[0], H=H2)
    error = periodicDist(pred, points_2d[0, :, 0])
    ax = fig.add_subplot(211)
    z = points_2d[0, :, 0]
    ax.scatter(x, error, c="orange", marker='+', label="gt")
    ax.set_xlabel('rho laser')
    ax.set_ylabel('error')

    ax = fig.add_subplot(212)
    z = points_2d[0, :, 0]
    ax.scatter(y, error, c="orange", marker='+', label="gt")
    ax.set_xlabel('theta laser')
    ax.set_ylabel('error')

    result = is_function(x,y,z)
    print("Is z = f(x, y) a function?", result)

    fig = plt.figure("ERROR LASER")

    pred = projectPoint2Image(laser_point=laser_points[0], H=H2)
    reprojected_laser = projectImage2Point(pred, H2)
    ax = fig.add_subplot(111)
    ax.scatter(points_3d[0,:,0], points_3d[0,:,1], c="orange", marker='+', label="gt")
    ax.scatter(reprojected_laser[:,0], reprojected_laser[:,1], c="red", marker='+', label="reproj")
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.legend()
    plt.show()

    results = {
        "H2":H2
    }

    # Specify the file path where you want to save the dictionary
    # file_path = "laser2camera_map.pkl"
    #
    # # save dictionary to person_data.pkl file
    # with open(file_path, 'wb') as fp:
    #     pickle.dump(results, fp)
    #     print('Dictionary saved successfully to file')
    #
    # print(f"Laser to Camera calibration results saved to {file_path}")


