# Initialize an empty dictionary to store the data
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D

def projectFull(H, X, Y):

    w11, w12, w21, w22, w31, w32, t1, t2, t3 = H

    u = (fu * w11 + u0 * w31) * X + (fu * w12 + u0 * w32) * Y + fu * t1 + u0 * t3
    v = (fv * w21 + v0 * w31) * X + (fv * w22 + v0 * w32) * Y + fv * t2 + v0 * t3

    return u,v

data_dict = {}

# Read the data from the file
with open('back.txt', 'r') as file:
    lines = file.readlines()

    current_image = None
    current_data = []

    for line in lines:
        line = line.strip()

        if line.endswith('back.jpg'):
            # Store the previous image's data if it exists
            if current_image is not None:
                data_dict[current_image] = current_data

            # Set the current image and initialize data list
            current_image = line
            current_data = []
        else:
            # Parse and store the data
            x, y = map(int, line[1:-1].split(','))
            current_data.append((x, y))

# Store the data for the last image
if current_image is not None:
    data_dict[current_image] = current_data

# Initialize an empty dictionary to store the data
data_dict2 = {}

# Read the data from the file
with open('ABback.txt', 'r') as file:
    lines = file.readlines()

    current_image = None
    current_data = []

    for line in lines:
        line = line.strip()

        if line.endswith('back.jpg'):
            # Store the previous image's data if it exists
            if current_image is not None:
                data_dict2[current_image] = current_data

            # Set the current image and initialize data list
            current_image = line
            current_data = []
        elif line.endswith(')'):
            # Store the previous im:
            # Parse and store the data
            x, y = map(float, line[1:-1].split(','))
            current_data.append((x, y))
        elif line.endswith('0'):
            a, b = map(float, line.split(';')[:2])
            current_data.append((a, b))

# Store the data for the last image
if current_image is not None:
    data_dict2[current_image] = current_data


image_points = data_dict
laserAB = data_dict2

AB = []
laser = []
image_p = []

for image in image_points.keys():

    image_p.append(image_points[image][0])
    AB.append(laserAB[image][0])
    laser.append(laserAB[image][1])

image_p = np.array(image_p)
AB = np.array(AB)
laser = np.array(laser)

fu = 250.001420127782
fv = 253.955300723887
u0 = 239.731339559399
v0 = 246.917074981568

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the 3D scatter plot
ax.scatter(laser[:,0], laser[:,1], AB[:,0], c=AB[:,0], cmap='viridis', marker='o')

# Set labels for the axes
ax.set_xlabel('X Laser')
ax.set_ylabel('Y Laser')
ax.set_zlabel('A')

plt.show()

A_matrix = []

for (A, B), (xt, yt) in zip(AB, laser):
    # w11, w12, w21, w22, w31, w32, t1, t2, t3 = H

    row = [A * fu * xt, A * fu * yt, B * fv * xt, B * fv * yt, (A * u0 + B * v0) * xt, (A * u0 + B * v0) * yt, A * fu,
           B * fv, (A * u0 + B * v0)]
    A_matrix.append(row)

A_matrix = np.array(A_matrix)
# A_matrix = A_matrix[::5,:]
B_matrix = np.ones(A_matrix.shape[0])
observed_points = image_p
laser_points = laser
# print(A_matrix)
# print(B_matrix)

U, S, VT = np.linalg.svd(A_matrix, full_matrices=False)
# Calculate the pseudoinverse of A
Ainv = np.dot(VT.transpose(), np.dot(np.diag(S ** -1), U.transpose()))

x = np.matmul(Ainv, B_matrix)

# Use the least-squares method to solve for x
# x, residuals, rank, s = np.linalg.lstsq(A_matrix, B_matrix, rcond=None)

print("Results: ")
print(x)

# Initialize an array to store the squared errors
squared_errors = []

for laser_p, image_p in zip(laser_points, observed_points):
    proj_x, proj_y = projectFull(x, laser_p[0], laser_p[1])
    # print(proj_x, proj_y)
    squared_error = (proj_x - image_p[0]) ** 2 + (proj_y - image_p[1]) ** 2
    squared_errors.append(squared_error)

avg_err = 0
for (a,b), (u,v) in zip(AB, observed_points):

    error = np.sum(np.abs(a * u + b * v - 1)) / math.sqrt(
            np.sum(a ** 2 + b ** 2))  # Use np.abs() for element-wise absolute value
    avg_err += error

# Calculate the RMSE
rmse = np.sqrt(np.mean(squared_errors))
print("Root Mean Square Error (RMSE):", rmse)

print("avg err", avg_err / len(observed_points))