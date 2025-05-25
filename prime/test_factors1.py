import torch
import torch.nn as nn
import pandas as pd

# Configuration
trace_flag = True
input_dim = 20
output_dim = 10
hidden_dim = 100
num_hidden_layers = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Format helper: round and convert tensor to list of floats with 2 decimal places
def format_tensor(t):
    return [round(float(v), 2) for v in t.squeeze()]

# Define the model architecture
class DeepBinaryNet(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=100, output_dim=10, num_hidden_layers=10):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if trace_flag:
            print(f"🔍 Input: {format_tensor(x)}")
            for i, layer in enumerate(self.model):
                if isinstance(layer, nn.Linear):
                    w = layer.weight.data
                    b = layer.bias.data
                    z = x @ w.t() + b
                    print(f"\nLayer {i} (Linear):")

                    print("  🔧 Weights and Biases (first 5 neurons):")
                    for j in range(min(5, w.size(0))):  # show only first 5 neurons
                        w_row = format_tensor(w[j])
                        b_val = round(float(b[j]), 2)
                        print(f"    Neuron {j}: W = {w_row}, b = {b_val}")
                    
                    print(f"  Input to Neuron (Wx + b): {format_tensor(z)}")
                    x = z
                if isinstance(layer, nn.ReLU):
                    x = torch.relu(x)
                    print(f"  ReLU Output: {format_tensor(x)}")
                if isinstance(layer, nn.Sigmoid):
                    x = torch.sigmoid(x)
                    print(f"  Sigmoid Output: {format_tensor(x)}")
                if isinstance(layer, nn.Tanh):
                    x = torch.tanh(x)
                    print(f"  Tanh Output: {format_tensor(x)}")
            print("\n✅ Final Output:", format_tensor(x))
            return x
        else:
            return self.model(x)

# Convert integer to fixed-width binary vector
def int_to_binvec(n, width):
    return torch.tensor([int(b) for b in f"{n:0{width}b}"], dtype=torch.float32)

# Load model and weights
model = DeepBinaryNet(input_dim=input_dim, output_dim=output_dim).to(device)
model.load_state_dict(torch.load("deep_binary_net.pth", map_location=device))
model.eval()

# Load test data
df = pd.read_csv("test_factors_output.csv")

# Evaluate with tracing
with torch.no_grad():
    for i, row in df.iterrows():
        print(f"\n=== Example {i+1} ===")
        x_val = int(row["Number"]) % (2 ** input_dim)
        x = int_to_binvec(x_val, input_dim).unsqueeze(0).to(device)
        model(x)
        if i >= 2:  # Limit to first 3 samples for readability
            break

