# !/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import argparse
import os
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


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


def calibrate(folder_path, CHECKERBOARD, args=None):

    # Defining the dimensions of checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

    # Creating vector to store vectors of 3D points for each checkerboard image
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Creating vector to store vectors of 2D points for each checkerboard image
    _img_shape = None
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Defining the world coordinates for 3D points
    square_size = args.square_size  # Set the size of your squares here (in meters)

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = square_size * np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    # Get a list of image file names in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png", ".tif"))]

    cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)

    # Iterate through the image files
    for image_file in tqdm(image_files):

        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)

        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

            cv2.imshow('calibration', img)
            # cv2.imwrite(fname, img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    h, w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """

    # N_OK = len(objpoints)

    N_imm = len(objpoints)  # number of calibration images
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]

    retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("Camera Matrix \n:", K)
    print("Distortions \n:", D)
    print("total error: {}".format(mean_error / len(objpoints)))

    results = {"K": K, "dist": D, "rvecs": rvecs, "tvecs": tvecs, "rpj_error": mean_error / len(objpoints)}

    # Specify the file path where you want to save the dictionary
    file_path = "intrinsicsUHD_ball_indoor2_fisheye.pkl"

    # save dictionary to person_data.pkl file
    with open(file_path, 'wb') as fp:
        pickle.dump(results, fp)
        print('Dictionary saved successfully to file')

    print(f"Intrinsics calibration results saved to {file_path}")

    # Load the sample image
    image_file = image_files[0]
    image_path = os.path.join(folder_path, image_file)

    img = cv2.imread(image_path)
    DIM = img.shape[:2]

    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    #
    # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    undistorted_img = undistort(img, K, D[:, :4])

    cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Undistorted Image', 1920, 1080)

    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image', 1920, 1080)

    # Display the original and undistorted images
    cv2.imshow('Original Image', img)
    cv2.imshow('Undistorted Image', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Train a one-shot detector")

    # Add a command-line argument
    parser.add_argument('--size_w', type=int, help='Widht of the board in corners', default=5)
    parser.add_argument('--size_h', type=int, help='Height of the board in corners', default=6)
    parser.add_argument('--square_size', type=float, help='Size of a square in the chessboard', default=0.11)

    # Parse the command-line arguments
    args = parser.parse_args()

    CHECKERBOARD = (args.size_w, args.size_h)

    folder_path = "images_UHD_ball_indoor2"  # Replace with the actual path to your dataset

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        exit()

    file_path = "intrinsicsUHD_ball_indoor2_fisheye.pkl"

    # Check if the file exists
    if os.path.exists(file_path):
        while True:
            user_input = input(
                f"The file '{file_path}' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
            if user_input == 'y':
                # User chose to overwrite the file
                calibrate(folder_path, CHECKERBOARD, args)
                break
            elif user_input == 'n':
                # User chose not to overwrite, you can handle this case accordingly
                print(f"Using the existing {file_path} calibration file.")

                with open(file_path, 'rb') as file:
                    # Load the dictionary from the pickle file
                    camera_calib = pickle.load(file)

                K = camera_calib['K']
                D = camera_calib['dist']

                # Get a list of image file names in the folder
                image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png", ".tif"))]

                # Load the sample image
                image_file = image_files[0]
                image_path = os.path.join(folder_path, image_file)

                img = cv2.imread(image_path)

                points = np.expand_dims(np.array([[500, 500], [1000,1000]]), axis=0).astype(np.float32)

                undistorted_points = undistortPoints(points, K, D, img.shape[:2])

                print(undistorted_points)

                # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
                # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                undistorted_img = undistort(img, K, D[:, :4])

                cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Undistorted Image', 1920, 1080)

                cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Original Image', 1920, 1080)

                # Display the original and undistorted images
                cv2.imshow('Original Image', img)
                cv2.imshow('Undistorted Image', undistorted_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                break
            else:
                print("Invalid input. Please enter 'y' to overwrite or 'n' to use the existing file.")
    else:
        # The file does not exist, proceed with your code
        print(f"The file {file_path} does not exist. Proceeding with your code.")
        calibrate(folder_path, CHECKERBOARD)


if __name__ == '__main__':
    main()
