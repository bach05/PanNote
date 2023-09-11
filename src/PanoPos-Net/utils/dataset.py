import torch
from torch.utils.data import Dataset
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module


class PanoPosDataset(Dataset):
    def __init__(self, file_list, image_res, mode="train", split_ratio=0.8, seed=None):
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

        for file in self.file_list:
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
                        box[0] = box[0] / self.img_res[0]
                        box[2] = box[2] / self.img_res[0]
                        box[1] = box[1] / self.img_res[1]
                        box[3] = box[3] / self.img_res[1]

                        self.data_list.append((box, centroid))

        # Shuffle the data randomly
        # random.shuffle(self.data_list)
        split_train = list(range(0, len(self.data_list), 4)) + list(range(1, len(self.data_list), 4)) + list(range(2, len(self.data_list), 4))
        split_train.sort()
        split_val = list(range(3, len(self.data_list), 4))

        # Split the data_list into train and val based on split_ratio
        #split_index = int(split_ratio * len(self.data_list))
        self.train_data = np.array(self.data_list)[split_train]
        self.val_data = np.array(self.data_list)[split_val]
        self.test_data = np.array(self.data_list)

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

        box, pos_2d = data

        box = torch.tensor(box)
        pos_2d = torch.tensor(pos_2d)

        return (box, pos_2d)


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
        ax.scatter(bbox_centers[:,0], bbox_centers[:,1], distances)

        # Set axis labels
        ax.set_xlabel('X box')
        ax.set_ylabel('Y box')
        ax.set_zlabel('2D distance')

        # Show the plot
        plt.show()

        print("v")

