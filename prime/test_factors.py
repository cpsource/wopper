import torch
import torch.nn as nn
import pandas as pd

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Define the same network architecture as during training
class DeepBinaryNet(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=100, output_dim=10, num_hidden_layers=10):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Convert an integer to a binary vector of fixed width
def int_to_binvec(n, width):
    return torch.tensor([int(b) for b in f"{n:0{width}b}"], dtype=torch.float32)

# Parameters
input_dim = 20
output_dim = 10

# Load test dataset
test_df = pd.read_csv("test_factors_output.csv")

# Load model and weights
model = DeepBinaryNet(input_dim=input_dim, output_dim=output_dim).to(device)
model.load_state_dict(torch.load("deep_binary_net.pth", map_location=device))
model.eval()
print("✅ Model loaded and ready for testing.")

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for _, row in test_df.iterrows():
        x_val = int(row["Number"]) % (2 ** input_dim)
        y_val = int(row["First Prime"]) % (2 ** output_dim)

        x = int_to_binvec(x_val, input_dim).unsqueeze(0).to(device)
        y_true = int_to_binvec(y_val, output_dim).to(device)

        y_pred = model(x).squeeze().round()

        print(f"y_pred {y_pred}, y_true = {y_true}")
        
        if torch.equal(y_pred, y_true):
            correct += 1
        total += 1

accuracy = correct / total * 100
print(f"\n📊 Test Set Accuracy: {accuracy:.2f}%")

