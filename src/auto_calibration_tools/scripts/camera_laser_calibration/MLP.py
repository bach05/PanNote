import torch
import torch.nn as nn
import torch.optim as optim


class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomMLP, self).__init__()
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList()

        # Create hidden layers based on the list of hidden sizes
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            self.hidden_layers.append(nn.LeakyReLU(negative_slope=0.1))
            input_size = hidden_size

        self.output_layer = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class MLPTrainer:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomMLP(input_size, hidden_sizes, output_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)

    def train(self, X, y, epochs, batch_size):

        X = torch.tensor(X).to(torch.float32)
        y = torch.tensor(y).to(torch.float32)

        X, y = X.to(self.device), y.to(self.device)
        dataset_size = X.size(0)

        for epoch in range(epochs):

            avg_loss = 0
            for i in range(0, dataset_size, batch_size):
                # Get a mini-batch of data
                inputs = X[i:i + batch_size]
                targets = y[i:i + batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                avg_loss += loss
                self.optimizer.step()

            # Compute statistics for weights and gradients
            weight_std = 0
            weight_avg = 0
            grad_avg = 0
            num_params = 0

            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    weight_std += param.data.std().item()
                    weight_avg += param.data.mean().item()
                    grad_avg += param.grad.mean().item()
                    num_params += 1


            weight_std /= num_params
            weight_avg /= num_params
            grad_avg /= num_params

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss.item() / (dataset_size // batch_size):.4f}')
            print(f'Weight Avg: {weight_avg:.4f}, Weight Std: {weight_std:.4f}, 'f'Gradient Avg: {grad_avg:.4f}')

    def predict(self, X):

        X = torch.tensor(X).to(torch.float32)
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
        return outputs

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


# Example usage
if __name__ == '__main__':
    # Sample data (replace with your own data)
    X_train = torch.randn(100, 3)  # Training input features (100 samples, 3 features)
    y_train = torch.randint(0, 2, (100, 1), dtype=torch.float32)  # Training binary labels

    input_size = X_train.shape[1]
    hidden_sizes = [64, 32]  # List of hidden layer sizes
    output_size = 1
    learning_rate = 0.01
    epochs = 100

    trainer = MLPTrainer(input_size, hidden_sizes, output_size, learning_rate)
    trainer.train(X_train, y_train, epochs)

    # Save the trained model
    trainer.save_model('trained_model.pth')

    # Example for loading and using the saved model
    loaded_trainer = MLPTrainer(input_size, hidden_sizes, output_size, learning_rate)
    loaded_trainer.load_model('trained_model.pth')

    # Sample data for prediction (replace with your own data)
    X_test = torch.randn(10, 3)  # Test input features (10 samples, 3 features)

    # Make predictions using the loaded model
    predictions = loaded_trainer.predict(X_test)
    print(predictions)
