import copy
import csv
from itertools import combinations

import cv2
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt, patches
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.auto_labeling_tools.util.laser_detector import convert_to_coordinates
import os
from tqdm import tqdm

import matplotlib as mpl


# Define your MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        """
        Initialize an MLP with arbitrary layers and sizes.

        Args:
            input_dim (int): Dimensionality of the input data.
            layer_sizes (list): List of integers specifying the sizes of each hidden layer.
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes

        # Create a list of linear layers and activation functions
        layers = []
        for i in range(len(layer_sizes)):
            if i == 0:
                # Input layer
                layers.append(nn.Linear(input_dim, layer_sizes[i]))
            else:
                # Hidden layers
                layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

            layers.append(nn.LeakyReLU())


        layers.append(nn.Linear(layer_sizes[-1], 2))

        # Combine the layers into a sequential model
        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        return self.mlp(x)

def read_scan(path):
    scan = []
    with open(path, 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Read each row of the CSV file
        image_ids = []
        for row in reader:
            image_id = int(row[0])  # Extract the image ID from the first column
            ranges = [float(value) for value in row[1].split(",")]  # Extract th
            scan.append(ranges)
            image_ids.append(image_id)

    return np.array(scan), image_ids


import torch
from torch.utils.data import Dataset
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module


def read_file_boxes(file_list, img_res):
    data_list = []
    for file in file_list:
        with open(file, "r") as ann_file:
            for line in ann_file:
                s = line.split(";")
                image = s[0]
                data = s[1:]
                for i in range(int(len(data) / 6)):
                    person = data[i * 6:(i + 1) * 6]
                    box = person[:4]
                    centroid = person[4:]

                    box = np.array(box).astype(np.float32)
                    centroid = np.array(centroid).astype(np.float32)

                    # Normalize box
                    box[0] = box[0] / img_res[0]
                    box[2] = box[2] / img_res[0]
                    box[1] = box[1] / img_res[1]
                    box[3] = box[3] / img_res[1]

                    data_list.append((box, centroid, [file, image]))
    return data_list

def read_file_skeletons(file_list, img_res):
    data_list = []

    x_positions = list(range(0, 17*3, 3))
    y_positions = list(range(1, 17*3, 3))
    z_positions = list(range(2, 17*3, 3))

    for file in file_list:
        with open(file, "r") as ann_file:
            for line in ann_file:
                s = line.split(";")
                image = s[0]
                data = s[1:]
                for i in range(int(len(data) / 57)):
                    person = data[i * 57:(i + 1) * 57]
                    box = person[:4]
                    centroid = person[4:6]

                    box = np.array(box).astype(np.float32)
                    centroid = np.array(centroid).astype(np.float32)

                    # Normalize box
                    box[0] = box[0] / img_res[0]
                    box[2] = box[2] / img_res[0]
                    box[1] = box[1] / img_res[1]
                    box[3] = box[3] / img_res[1]

                    sk = np.array(person[6:]).astype(np.float32)

                    '''
                    x = sk[x_positions] / img_res[0]
                    y = sk[y_positions] / img_res[1]
                    c = sk[z_positions]

                    sk = np.array([x,y,c])
                    '''

                    data_list.append((sk, centroid, image))
    return data_list


class PanoPosDataset(Dataset):
    def __init__(self, file_list, image_res, mode="train", split_ratio=0.8, seed=None, skeleton=False):
        """
        Initialize the CustomDataset with a list of file paths.

        Args:
            file_list (list): A list of file paths containing your data.
            image_res (tuple): Resolution of the images (width, height).
            split_ratio (float): Ratio of data to use for training (0.0 to 1.0).
            seed (int): Seed for reproducible random splitting.
        """
        self.file_list = file_list
        self.img_res = image_res
        self.split_ratio = split_ratio
        assert mode in ["train", "val", "test"]
        self.mode = mode

        # Seed the random number generator for reproducibility
        if seed is not None:
            random.seed(seed)

        self.data_list = []  # the list contains tuples [box, centroid]

        if skeleton:
            self.data_list = read_file_skeletons(self.file_list, self.img_res)
        else:
            self.data_list = read_file_boxes(self.file_list, self.img_res)

        # Shuffle the data randomly
        # random.shuffle(self.data_list)
        split_train = list(range(0, len(self.data_list), 4)) + list(range(1, len(self.data_list), 4)) + list(range(2, len(self.data_list), 4))
        split_train.sort()
        split_val = list(range(3, len(self.data_list), 4))

        # Split the data_list into train and val based on split_ratio
        #split_index = int(split_ratio * len(self.data_list))
        self.train_data = [self.data_list[k] for k in split_train]
        self.val_data = [self.data_list[k] for k in split_train]
        self.test_data =[self.data_list[k] for k in range(len(self.data_list))]

        if self.mode == "train":
            self.data = self.train_data
        elif self.mode == "val":
            self.data = self.train_data
        elif self.mode == "test":
            self.data = self.test_data


    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)
    def __getitem__(self, idx):
        """
        Load and return data from the file at the given index.

        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            data (tuple of torch.Tensor): The loaded data as a tuple of PyTorch tensors (box, pos_2d).
        """
        data = self.data[idx]

        box, pos_2d, im = data

        box = torch.tensor(box)
        pos_2d = torch.tensor(pos_2d)

        return (box, pos_2d, im)


    def visualize_data(self):

        bbox_centers = []
        distances = []

        for box, pos in self.data_list:

            center = (box[0] + box[2]/2, box[1] + box[3]/2)
            bbox_centers.append(center)
            d = np.linalg.norm(pos)
            distances.append(d)

        bbox_centers = np.array(bbox_centers)
        distances = np.array(distances)

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D scatter plot
        ax.scatter(bbox_centers[:, 0], bbox_centers[:, 1], distances)

        # Set axis labels
        ax.set_xlabel('X box')
        ax.set_ylabel('Y box')
        ax.set_zlabel('2D distance')

        # Show the plot
        plt.show()

        print("v")

if __name__ == "__main__":

    input_dim = 4  # Change this to match your input dimension
    layer_sizes = [8, 16, 32, 64, 128]  # Specify the sizes of hidden layers
    learning_rate = 0.0005
    batch_size = 4
    epochs = 200

    laser_spec = {
        'frame_id': "base_link",
        'angle_min': -3.140000104904175,
        'angle_max': 3.140000104904175,
        'angle_increment': 0.005799999926239252,
        'range_min': 0.44999998807907104,
        'range_max': 25.0,
        'len': 1083
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path_out = "./save_images/"
    if not os.path.exists(base_path_out):
        os.makedirs(base_path_out)
        print(f"Folder '{base_path_out}' created.")
    else:
        print(f"Folder '{base_path_out}' already exists.")

    # Initialize your custom dataset
    base_path = "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/bag_extraction"
    files = ['hospital3_static', 'lab_indoor_1', 'lab_indoor_3_2', 'lab_outdoor_1_2']
    file_test = ['lab_indoor_3_2']

    out_path = os.path.join(base_path_out,file_test[0])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Folder '{out_path}' created.")
    else:
        print(f"Folder '{out_path}' already exists.")

    path_model = f"./saved_models/full/MLP_train_testFold-{file_test[0]}.pth"
    model = MLP(input_dim, layer_sizes).to(device)
    model.load_state_dict(torch.load(path_model))
    model.cuda()

    file_list_test_auto = [os.path.join(base_path, file, "out", "automatic_annotations.csv") for file in file_test]
    file_list_test_man = [os.path.join(base_path, file, "annotations.csv") for file in file_test]

    test_dataset = PanoPosDataset(file_list_test_man, image_res=[3840, 1920], mode="test")
    test_dataset_auto = PanoPosDataset(file_list_test_auto, image_res=[3840, 1920], mode="test")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader_auto = DataLoader(test_dataset_auto, batch_size=1, shuffle=False, num_workers=0)
    scans = None
    boxes = []
    boxes_auto = []
    poses = []
    poses_auto = []
    images = []
    images_auto = []
    with torch.no_grad():
        for box, pos_2d, img in (test_loader):
            boxes.append(box.numpy())
            poses.append(pos_2d.numpy())
            images.append(img[1][0])
            box = box.to(device)
            train_pos_2d = pos_2d.to(device)

    path = img[0][0]
    path = path[:path.rindex('/')]

    with torch.no_grad():
        for box, pos_2d, img in (test_loader_auto):
            boxes_auto.append(box.numpy())
            poses_auto.append(pos_2d.numpy())
            images_auto.append(img[1][0])
            box = box.to(device)
            train_pos_2d = pos_2d.to(device)

    boxes = np.array(boxes)
    boxes_auto = np.array(boxes_auto)
    poses = np.array(poses)
    poses_auto = np.array(poses_auto)
    images = np.array(images)
    images_auto = np.array(images_auto)


    scans, ids = read_scan(path + "/laser.csv")

    for image_n in tqdm(np.unique(images)):
        image_path = path + "/img/" + image_n+ ".png"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        scan_index = ids.index(int(image_n))
        scan_for_image = scans[scan_index]
        points, polar = convert_to_coordinates(scan_for_image, laser_spec, remove_out=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

        # Plot the image in the first subplot
        ax1.imshow(image)
        ax1.axis('off')  # Turn off axis labels and ticks
        # Turn off tick labels
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.xaxis.set_ticks_position('none')
        ax2.yaxis.set_ticks_position('none')
        boxes_image = boxes[np.where(images == image_n)]
        poses_image = poses[np.where(images == image_n)]

        boxes_image_auto = boxes_auto[np.where(images_auto == image_n)]
        poses_image_auto = poses_auto[np.where(images_auto == image_n)]
        poses_image_auto = poses_image_auto[:, 0]
        poses_image = poses_image[:, 0]

        M = np.max(np.concatenate((poses_image, poses_image_auto)), axis=0)
        m = np.min(np.concatenate((poses_image, poses_image_auto)), axis=0)



        total_boxes = np.concatenate((boxes_image, boxes_image_auto))
        total_boxes = np.unique(total_boxes, axis=0)

        #colors = mpl.cm.rainbow(np.linspace(0, 1, len(total_boxes)))
        colors = ["#f55657", "#f5c538", "#83c124", "#1a7abb", "#684a8c", "#b09fc0"]
        ms = 30
        lw = 5
        ax2.scatter(points[:, 0], points[:, 1], marker=".", color="black", s=20, alpha=0.5)


        for bn, box in enumerate(total_boxes):
            index_auto = np.where((boxes_image_auto[:, 0, 0] == box[0, 0]) &
                     (boxes_image_auto[:, 0, 1] == box[0, 1]) &
                     (boxes_image_auto[:, 0, 2] == box[0, 2]) &
                     (boxes_image_auto[:, 0, 3] == box[0, 3]))

            if len(index_auto[0]) > 0:
                index_auto = index_auto[0][0]

                #ax2.scatter(poses_image_auto[index_auto, 0], poses_image_auto[index_auto, 1], marker="^", color=colors[bn], s=80, alpha=0.75)
                #obj = ax2.scatter(poses_image_auto[index_auto, 0], poses_image_auto[index_auto, 1], marker="^", color=None, s=80, alpha=1, edgecolors='b')
                #obj.set_facecolor=("none")
                ax2.plot(poses_image_auto[index_auto, 0], poses_image_auto[index_auto, 1], '_', ms=ms, markerfacecolor=(0,0,0,0), markeredgecolor=colors[bn],  markeredgewidth=lw)
            index_man = np.where((boxes_image[:, 0, 0] == box[0, 0]) &
                                  (boxes_image[:, 0, 1] == box[0, 1]) &
                                  (boxes_image[:, 0, 2] == box[0, 2]) &
                                  (boxes_image[:, 0, 3] == box[0, 3]))

            if len(index_man[0]) > 0:
                index_man = index_man[0][0]
                #ax2.scatter(poses_image[index_man, 0], poses_image[index_man, 1], marker="s", color=colors[bn], s=80, alpha=0.75)
                #obj = ax2.scatter(poses_image[index_man, 0], poses_image[index_man, 1], marker="s", color=None, s=80, alpha=1, edgecolors='b')
                #obj.set_facecolor=("none")
                ax2.plot(poses_image[index_man, 0], poses_image[index_man, 1], '|', ms=ms, markerfacecolor=colors[bn], markeredgecolor=colors[bn], markeredgewidth=lw)

            top_left = box[0, :2] * [3840, 1920]
            bottom_right = box[0, 2:] * [3840, 1920]

            width = bottom_right[0] - top_left[0]
            height = top_left[1] - bottom_right[1]

            top_left[1] = bottom_right[1]

            rectangle = patches.Rectangle(top_left, width, height, linewidth=3, edgecolor=colors[bn], facecolor='none')
            ax1.add_patch(rectangle)

            torch_box = torch.tensor(box).cuda()
            prediction = model(torch_box).detach().cpu().numpy()[0]
            ax2.plot(prediction[0], prediction[1], 'x', ms=ms, markerfacecolor="blue", markeredgecolor=colors[bn], markeredgewidth=lw-1)
        #plt.setp(ax2, xlim=[m[0], M[0]], ylim=[m[1], M[1]])
        ax2.set_xlim(m[0]-1, M[0]+1)
        ax2.set_ylim(m[1]-1, M[1]+1)
        plt.subplots_adjust(left=0, bottom=0.05, right=0.98, top=0.90, wspace=0.03, hspace=None)



        # Plot the laser scan data in the second subplot

        #ax2.scatter(poses_image_auto[:, 0], poses_image_auto[:, 1], marker="s", color="red", s=6)



        #ax2.set_title("Laser Scan Data")
        #ax2.axis('equal')
        # Plot horizontal x-axis (red line)
        #ax2.set_facecolor('#D3D3D3')
        ax2.grid()
        #plt.show()

        plt.savefig(os.path.join(out_path, image_n+".png"))
        plt.close()

    # Add the rectangle patch to the axis

