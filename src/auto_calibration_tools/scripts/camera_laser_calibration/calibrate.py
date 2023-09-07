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

import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

def project(H, xt, yt):

    a0, b0, c0, a1, b1, c1 = H

    u = a0 * xt + b0 * yt + c0
    v = a1*xt + b1 * yt + c1

    return u,v


def projectFull(H, X, Y):

    w11, w12, w21, w22, w31, w32, t1, t2, t3 = H

    u = (fu * w11 + u0 * w31) * X + (fu * w12 + u0 * w32) * Y + fu * t1 + u0 * t3
    v = (fv * w21 + v0 * w31) * X + (fv * w22 + v0 * w32) * Y + fv * t2 + v0 * t3

    return u,v


def projectVLINE(H, X, Y):
    # w11, w12, w31, w32, t1, t3 = H

    w11, w12, w31, w32, t1, t3 = H

    u = (fu * w11 + u0 * w31) * X + (fu * w12 + u0 * w32) * Y + fu * t1 + u0 * t3

    return u

def distErr(H, A, xt, yt):

    u = projectVLINE(H, xt, yt)

    error = np.sum((A * u - 1)**2)

    return error

def optimize(H0, A, xt, yt, error_function):

    optimization_function = lambda H: error_function(H, A, xt, yt)
    result = minimize(optimization_function, H0, method='Nelder-Mead')

    optimized_H = result.x
    optimized_error = result.fun

    print("Optimization res: ", result.success)
    print("Message: ", result.message)

    return optimized_H, optimized_error


def LSoptimize(H0, A, xt, yt, error_function):

    optimization_function = lambda H: error_function(H, A, xt, yt)
    options = {'xtol': 1e-6, 'ftol': 1e-6, 'maxfev': 1000}
    result = least_squares(optimization_function, H0, method='trf', xtol=options['xtol'], ftol=options['ftol'],
                        max_nfev=options['maxfev'])

    optimized_H = result.x
    optimized_error = result.fun

    print("Optimization res: ", result.success)
    print("Message: ", result.message)

    return optimized_H, optimized_error


def calculate_coefficients(u0,v0,u1,v1):
    ab = (u0 - v0) / (u1 - v1)
    cb = -ab * u0 + v0
    B = 1 / cb
    A = -ab / cb
    return A,B

if __name__ == '__main__':

    ####### CAMERA MATRIX (TRIAL)
    calib_file = "../calibration_data_intrinsics/intrinsicsUHD.pkl"
    # Open the pickle file for reading in binary mode
    with open(calib_file, 'rb') as file:
        # Load the dictionary from the pickle file
        camera_calib = pickle.load(file)

    # Specify the path of the calibration data
    file_path = "cameraLaser_pointsUHD_indoor.pkl"

    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the dictionary from the pickle file
        data = pickle.load(file)

    for side, points in data.items():

        # points = list of tuples like ( (C, left),(B, right) )
        # left/right numpy array 5x2

        print("*****Processing side {}".format(side))
        print("*****Found {} tuples".format(len(points)))

        parameters = []
        observed_points = []
        laser_points = []
        names = []

        for point in points:

            a, b, img_name = point

            names.append(img_name)
            names.append(img_name)

            p = (a,b)

            #tuple = a

            for tuple in p:

                laser_point, board_points = tuple

                u_coors = board_points[:, 0]  # Extracts the first column (u coordinates)
                v_coords = board_points[:, 1]  # Extracts the second column (v coordinates)
                X = laser_point[0]
                Y = laser_point[1]

                observed_points.append(np.array([u_coors[-1], v_coords[-1]]))
                #laser_points.append( np.tile(np.array([X,Y]),(5,1)) )
                laser_points.append( np.array([X,Y]) )


                A1, B1 = calculate_coefficients( u_coors[0], v_coords[0],  u_coors[-1], v_coords[-1])

                param = (A1, B1, X, Y)
                parameters.append(param)

            # A2, B2 = calculate_coefficients( u_coors[1], v_coords[1],  u_coors[-2], v_coords[-2])
            # param = (A2, B2, X, Y)
            # #print("B: {}".format(B2))
            # parameters.append(param)


        #Start Optimization
        #random.shuffle(parameters)
        parameters = np.array(parameters)
        k = camera_calib[side]['K']

        # # Create a 3D figure
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # # Create the 3D scatter plot
        # ax.scatter(parameters[:,1], parameters[:,2], parameters[:,0], c=parameters[:,0], cmap='viridis', marker='o')
        #
        # # Set labels for the axes
        # ax.set_xlabel('X Laser')
        # ax.set_ylabel('Y Laser')
        # ax.set_zlabel('A')
        #
        # plt.show()

        fu = k[0,0]
        fv = k[1,1]
        u0 = k[0,2]
        v0 = k[1,2]

        A_matrix = []


        # for A,B,xt,yt in parameters:
        #
        #     row = [ A*xt, A*yt, 1, B*xt, B*yt, 1 ]
        #     A_matrix.append(row)


        for A,B,xt,yt in parameters:

            #w11, w12, w21, w22, w31, w32, t1, t2, t3 = H

            row = [A*fu*xt,  A*fu*yt,  B*fv*xt,  B*fv*yt, (A*u0 + B*v0)*xt, (A*u0 + B*v0)*yt,  A*fu, B*fv, (A*u0 + B*v0)]
            A_matrix.append(row)

        # for A, xt, yt in parameters[:,:-1]:
        #     # w11, w12, w31, w32, t1, t3 = H
        #
        #     row = [A * fu * xt, A * fu * yt, (A * u0) * xt, (A * u0) * yt, A * fu, (A * u0)]
        #     A_matrix.append(row)

        A_matrix = np.array(A_matrix)
        #A_matrix = A_matrix[::5,:]
        B_matrix = np.ones(A_matrix.shape[0])
        observed_points = np.array(observed_points)
        laser_points = np.array(laser_points)
        #print(A_matrix)
        #print(B_matrix)

        U, S, VT = np.linalg.svd(A_matrix, full_matrices=False)

        #r = 0.000001 + 0.000005 * np.random.rand(S.shape[0])
        #S  = S + r

        #r = 0.001 + 0.005 * np.random.rand(B_matrix.shape[0])
        #B_matrix = B_matrix + r

        # Calculate the pseudoinverse of A
        #Ainv= np.dot(VT.transpose(),np.dot(np.diag(S**-1),U.transpose()))

        #r = 0.000001 + 0.000005 * np.random.rand(A_matrix.shape[1], A_matrix.shape[0])
        Ainv = np.linalg.pinv(A_matrix)


        H = np.matmul(Ainv, B_matrix)

        # r = 0.001 + 0.005 * np.random.rand(H.shape[0])
        # H = H + r


        # Use the least-squares method to solve for x
        #x, residuals, rank, s = np.linalg.lstsq(A_matrix, B_matrix, rcond=None)

        print("Results Initiliazation: ")
        print(H)

        #OPTIMIZE INITIAL GUESS
        # H2, err = optimize(H, parameters[:,0], parameters[:,1], parameters[:,2], error_function=distErr)
        #
        # print("Results Optimization: ")
        # print(H2)
        # print("Error: {}".format(err))


        #TRY MLP
        # mean = np.mean(parameters[:,1:3], axis = 0)
        # std = np.std(parameters[:,1:3], axis = 0)
        # X_train = ((parameters[:,1:3])  - mean) / std
        # input_size = X_train.shape[1]
        # hidden_sizes = [8, 16, 16, 8, 4]  # List of hidden layer sizes
        # output_size = 1
        # learning_rate = 0.01
        # epochs = 100
        #
        # mean_y = np.mean(parameters[:,-1])
        # std_y = np.std(parameters[:,-1])
        # y_train = (parameters[:,-1] - mean_y) / std_y
        #
        # trainer = MLPTrainer(input_size, hidden_sizes, output_size, learning_rate)
        # trainer.train(X_train, y_train, epochs, batch_size=16)

        # Initialize an array to store the squared errors
        squared_errors = []
        squared_errors_MLP = []

        for i, (laser_p, image_p) in enumerate(zip(laser_points, observed_points)):

            name = names[i]

            # im_name = "cropped_" + os.path.basename(name) + ".jpg"
            # im_root = "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration/"
            # img_name = os.path.join(im_root, side, im_name)
            # img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

            proj_x, proj_y = projectFull(H, laser_p[0], laser_p[1])

            # input = (laser_p - mean) / std
            # pred = trainer.predict(input)
            #
            # pred = (pred.detach().cpu().numpy() * std_y) + mean_y

            # plt.imshow(img)
            # plt.axvline(proj_x, color="red")
            # plt.axvline(pred, color="cyan")
            # plt.scatter(image_p[0], image_p[1], color="blue", marker="x")

            squared_error = (proj_x - image_p[0]) ** 2
            squared_errors.append(squared_error)

            #print(f"Xp {proj_x} : X {image_p[0]},Y {image_p[1]} ")
            #plt.show()

        # Calculate the RMSE
        rmse = np.sqrt(np.mean(squared_errors))
        print(" \033[1m Root Mean Square Error (RMSE): {} \033[0m".format(rmse))

        rmse = np.sqrt(np.mean(squared_errors_MLP))
        print(" \033[1m [MLP] Root Mean Square Error (RMSE): {} \033[0m".format(rmse))

        avg_err = 0
        # for (a, b), (xt,yt) in zip(parameters[:,:2], laser_points):
        #
        #     u, v = projectFull(x, xt, yt)
        #     error = np.sum(np.abs(a * u + b * v - 1)) / math.sqrt(
        #         np.sum(a ** 2 + b ** 2))  # Use np.abs() for element-wise absolute value
        #     avg_err += error
        #
        # print("avg err", avg_err / len(observed_points))





