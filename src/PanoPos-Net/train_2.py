import copy
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import PanoPosDataset  # Replace with the actual name of your dataset class
import os

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

def train():

    input_dim = 4  # Change this to match your input dimension
    layer_sizes = [8, 16, 32, 64, 128]  # Specify the sizes of hidden layers
    learning_rate = 0.0005
    batch_size = 4
    epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize your custom dataset
    base_path = "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/bag_extraction"
    model_base_path = "./saved_models"

    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)
        print(f"Folder '{model_base_path}' created.")
    else:
        print(f"Folder '{model_base_path}' already exists.")

    files = ['hospital3_static', 'lab_indoor_1', 'lab_indoor_3_2', 'lab_outdoor_1_2']

    manual_tests = []
    auto_tests = []
    motion_tests = []

    for i in range(len(files)):
        file_test = [files[i]]
        file_train = []
        file_train_full = []
        for j in range(len(files)):
            if j != i:
                file_train.append(files[j])
                file_train_full.append(files[j]+"_full")


        print("Test: ", files[i])

        for rep in range(1):
            '''    
            file_train = [
                'hospital3_static',
                'lab_indoor_1',
                'lab_indoor_3_2',
            ]
            file_train_full = [
                'hospital3_static_full',
                'lab_indoor_1_full',
                'lab_indoor_3_2_full',
            ]
            file_test = [
                'lab_outdoor_1_2'
            ]
            '''

            file_list_auto = [os.path.join(base_path, file, "out", "automatic_annotations.csv") for file in file_train] #auto red
            file_list_man = [os.path.join(base_path, file, "annotations.csv") for file in file_train] #manual
            file_list_auto_full = [os.path.join(base_path, file, "out", "automatic_annotations.csv") for file in
                              file_train_full]  # auto full

            file_list_test_auto = [os.path.join(base_path, file, "out", "automatic_annotations.csv") for file in file_test]
            file_list_test_man = [os.path.join(base_path, file, "annotations.csv") for file in file_test]

            train_folder = "manual"
            model_save_path = os.path.join(model_base_path, train_folder)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
                print(f"Folder '{model_save_path}' created.")
            else:
                print(f"Folder '{model_save_path}' already exists.")

            train_dataset = PanoPosDataset(file_list_man, image_res=[3840, 1920], mode="train")
            val_dataset = PanoPosDataset(file_list_man, image_res=[3840, 1920], mode="val")

            test_dataset = PanoPosDataset(file_list_test_man, image_res=[3840, 1920], mode="test")
            test_dataset_auto = PanoPosDataset(file_list_test_auto, image_res=[3840, 1920], mode="test")

            file_list_test_motion = [os.path.join(base_path,"motion_lab1_small/motion_lab1.csv") ]
            test_dataset_motion = PanoPosDataset(file_list_test_motion, image_res=[3840, 1920], mode="test")

            #train_dataset.visualize_data()

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
            test_loader_auto = DataLoader(test_dataset_auto, batch_size=1, shuffle=True, num_workers=0)
            test_loader_motion = DataLoader(test_dataset_motion, batch_size=1, shuffle=True, num_workers=0)


            # Initialize the MLP model, loss function, and optimizer
            model = MLP(input_dim, layer_sizes).to(device)
            # print(model)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

            # Training loop
            best_loss = 100
            p = 0
            best_model = None
            for epoch in tqdm(range(epochs)):

                train_loss = 0
                for box, pos_2d in (train_loader):

                    box = box.to(device)
                    train_pos_2d = pos_2d.to(device)

                    # Zero the gradients
                    optimizer.zero_grad()
                    # Forward pass
                    train_outputs = model(box)

                    # Compute the loss
                    loss = criterion(train_outputs, train_pos_2d)
                    train_loss += loss

                    # Backpropagation
                    loss.backward()
                    optimizer.step()

                train_loss /= len(train_loader)


                # Validation
                model.eval()  # Set the model in evaluation mode (e.g., for batch normalization)

                val_loss = 0.0
                with torch.no_grad():
                    for val_box, val_pos_2d in val_loader:
                        val_box = val_box.to(device)
                        val_pos_2d = val_pos_2d.to(device)

                        val_outputs = model(val_box)
                        val_loss += criterion(val_outputs, val_pos_2d).item()

                val_loss /= len(val_loader)  # Calculate the average validation loss

                # Print the loss for this epoch and validation loss
                #print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')

                if val_loss < best_loss:
                    p = 0
                    best_loss = val_loss
                    best_model = copy.deepcopy(model)
                else:
                    p += 1

                if p > 5:
                    print(f"Early stopping at {epoch}")
                    break

            model = best_model

            model_name = f"MLP_train_testFold-{file_test[0]}.pth"
            model_path = os.path.join(model_save_path, model_name)
            torch.save(model.state_dict(), model_path)
            print(f"SAVED model in {model_path}")

            #save the model

            distances = []
            correct_1 = 0
            correct_05 = 0
            correct_025 = 0
            correct_075 = 0
            correct_010 = 0
            matched_boxes = 0
            with torch.no_grad():
                val_loss = 0.0
                for val_box, val_pos_2d in test_loader:
                    matched_boxes += 1
                    val_box = val_box.to(device)
                    val_pos_2d = val_pos_2d.to(device)

                    val_outputs = model(val_box)
                    val_loss += criterion(val_outputs, val_pos_2d).item()

                    dist = np.linalg.norm((np.array(val_outputs.cpu().numpy()).astype(float) - val_pos_2d.cpu().numpy()))
                    distances.append(dist)
                    if dist < 0.10:
                        correct_1 += 1
                        correct_05 += 1
                        correct_025 += 1
                        correct_010 += 1
                        correct_075 += 1
                    elif dist < 0.25:
                        correct_1 += 1
                        correct_05 += 1
                        correct_025 += 1
                        correct_075 += 1
                    elif dist < 0.5:
                        correct_1 += 1
                        correct_05 += 1
                        correct_075 += 1
                    elif dist < 0.75:
                        correct_1 += 1
                        correct_075 += 1
                    elif dist < 1:
                        correct_1 += 1

            val_loss /= len(val_loader)  # Calculate the average validation loss

            distances = np.array(distances)

            # Print the loss for this epoch and validation loss
            print(f'Test man Loss: {val_loss:.4f} MSE: {np.mean(distances):.4f} ACC 010:  {correct_010/matched_boxes:.4f} ACC 025:  {correct_025/matched_boxes:.4f} ACC 05:  {correct_05/matched_boxes:.4f}, ACC 075:  {correct_075/matched_boxes:.4f}')

            manual_tests.append( [np.mean(distances), correct_010/matched_boxes, correct_025/matched_boxes, correct_075/matched_boxes] )

            distances = []
            correct_1 = 0
            correct_05 = 0
            correct_025 = 0
            correct_010 = 0
            correct_075 = 0
            matched_boxes = 0
            with torch.no_grad():
                val_loss = 0.0
                for val_box, val_pos_2d in test_loader_auto:
                    matched_boxes += 1

                    val_box = val_box.to(device)
                    val_pos_2d = val_pos_2d.to(device)

                    val_outputs = model(val_box)
                    val_loss += criterion(val_outputs, val_pos_2d).item()

                    dist = np.linalg.norm((np.array(val_outputs.cpu().numpy()).astype(float) - val_pos_2d.cpu().numpy()))
                    distances.append(dist)
                    if dist < 0.10:
                        correct_1 += 1
                        correct_05 += 1
                        correct_025 += 1
                        correct_010 += 1
                        correct_075 += 1
                    elif dist < 0.25:
                        correct_1 += 1
                        correct_05 += 1
                        correct_025 += 1
                        correct_075 += 1
                    elif dist < 0.5:
                        correct_1 += 1
                        correct_05 += 1
                        correct_075 += 1
                    elif dist < 0.75:
                        correct_1 += 1
                        correct_075 += 1
                    elif dist < 1:
                        correct_1 += 1

            val_loss /= len(val_loader)  # Calculate the average validation loss
            distances = np.array(distances)

            # Print the loss for this epoch and validation loss
            print(f'Test auto Loss: {val_loss:.4f} MSE: {np.mean(distances):.4f} ACC 010:  {correct_010/matched_boxes:.4f} ACC 025:  {correct_025/matched_boxes:.4f} ACC 05:  {correct_05/matched_boxes:.4f} ACC 075:  {correct_075/matched_boxes:.4f}')

            auto_tests.append([np.mean(distances), correct_010 / matched_boxes, correct_025 / matched_boxes,
                                 correct_075 / matched_boxes])

            distances = []
            correct_1 = 0
            correct_05 = 0
            correct_025 = 0
            correct_010 = 0
            correct_075 = 0
            matched_boxes = 0
            with torch.no_grad():
                val_loss = 0.0
                for val_box, val_pos_2d in test_dataset_motion:
                    matched_boxes += 1

                    val_box = val_box.to(device)
                    val_pos_2d = val_pos_2d.to(device)

                    val_outputs = model(val_box)
                    val_loss += criterion(val_outputs, val_pos_2d).item()

                    dist = np.linalg.norm(
                        (np.array(val_outputs.cpu().numpy()).astype(float) - val_pos_2d.cpu().numpy()))
                    distances.append(dist)
                    if dist < 0.10:
                        correct_1 += 1
                        correct_05 += 1
                        correct_025 += 1
                        correct_010 += 1
                        correct_075 += 1
                    elif dist < 0.25:
                        correct_1 += 1
                        correct_05 += 1
                        correct_025 += 1
                        correct_075 += 1
                    elif dist < 0.5:
                        correct_1 += 1
                        correct_05 += 1
                        correct_075 += 1
                    elif dist < 0.75:
                        correct_1 += 1
                        correct_075 += 1
                    elif dist < 1:
                        correct_1 += 1

            val_loss /= len(val_loader)  # Calculate the average validation loss
            distances = np.array(distances)

            # Print the loss for this epoch and validation loss
            print(
                f'Test man MOTION Loss: {val_loss:.4f} MSE: {np.mean(distances):.4f} ACC 010:  {correct_010 / matched_boxes:.4f} ACC 025:  {correct_025 / matched_boxes:.4f} ACC 05:  {correct_05 / matched_boxes:.4f} ACC 075:  {correct_075/matched_boxes:.4f}')

            motion_tests.append([np.mean(distances), correct_010 / matched_boxes, correct_025 / matched_boxes,
                                 correct_075 / matched_boxes])


    print("*********** Summary k-fold *************")
    manual_tests = np.array(manual_tests)
    print(manual_tests)
    manual_tests = manual_tests.mean(axis=0)
    print(f" Manual: mse {manual_tests[0]}, acc 0.1 {manual_tests[1]}, acc 0.25 {manual_tests[2]}, acc 0.75 {manual_tests[3]}")

    auto_tests = np.array(auto_tests)
    auto_tests = auto_tests.mean(axis=0)
    print(
        f" Automatic: mse {auto_tests[0]}, acc 0.1 {auto_tests[1]}, acc 0.25 {auto_tests[2]}, acc 0.75 {auto_tests[3]}")

    motion_tests = np.array(motion_tests)
    motion_tests = motion_tests.mean(axis=0)
    print(
        f" Motion: mse {motion_tests[0]}, acc 0.1 {motion_tests[1]}, acc 0.25 {motion_tests[2]}, acc 0.75 {motion_tests[3]}")


if __name__ == "__main__":
    train()
