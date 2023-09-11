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
    layer_sizes = [4, 8, 16, 32, 64]  # Specify the sizes of hidden layers
    learning_rate = 0.0005
    batch_size = 4
    epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize your custom dataset
    base_path = "/home/iaslab/ROS_AUTOLABELLING/AutoLabeling/src/auto_calibration_tools/bag_extraction"
    #files = ['lab_indoor_1/annotations_lab_indoor_1.csv', 'hospital3_static/annotations_hospital3_static.csv']  # Replace with your file paths
    files = ['/home/leonardo/Downloads/labelling_csv/annotations_h1.csv', '/home/leonardo/Downloads/labelling_csv/annotations_h3.csv', '/home/leonardo/Downloads/labelling_csv/annotations_l1.csv']
    file_list = [os.path.join(base_path, file) for file in files]

    train_dataset = PanoPosDataset(file_list, image_res=[3840, 1920], mode="train")
    val_dataset = PanoPosDataset(file_list, image_res=[3840, 1920], mode="val")

    #train_dataset.visualize_data()

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)


    # Initialize the MLP model, loss function, and optimizer
    model = MLP(input_dim, layer_sizes).to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Training loop
    for epoch in range(epochs):

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
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}')



if __name__ == "__main__":
    train()
