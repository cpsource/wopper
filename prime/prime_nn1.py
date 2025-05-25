import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Define the deep neural network
class DeepBinaryNet(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=100, output_dim=10, num_hidden_layers=10):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Step 2: Convert an integer to a fixed-width binary vector
def int_to_binvec(n, width):
    return torch.tensor([int(b) for b in f"{n:0{width}b}"], dtype=torch.float32)

# Step 3: Load and split the dataset
data = pd.read_csv("prime_factors_output.csv")
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

# Step 4: Setup model, loss, optimizer
input_dim = 20
output_dim = 10
model = DeepBinaryNet(input_dim=input_dim, output_dim=output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Step 5: Training loop
epochs = 20
for epoch in range(epochs):
    print(f"Epoch Start = {epoch}")
    
    model.train()
    total_loss = 0
    for _, row in train_df.iterrows():
        # Convert input number to fixed-width binary vector
        x_val = int(row["Number"])
        if x_val >= 2 ** input_dim:
            x_val = x_val % (2 ** input_dim)
        x = int_to_binvec(x_val, input_dim).unsqueeze(0)

        # Convert target number to fixed-width binary vector
        y_val = int(row["First Prime"])
        if y_val >= 2 ** output_dim:
            y_val = y_val % (2 ** output_dim)
        y = int_to_binvec(y_val, output_dim).unsqueeze(0)

        # Forward, backward, optimize
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Step 6: Evaluate on validation set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for _, row in val_df.iterrows():
        x_val = int(row["Number"])
        if x_val >= 2 ** input_dim:
            x_val = x_val % (2 ** input_dim)
        x = int_to_binvec(x_val, input_dim).unsqueeze(0)

        y_val = int(row["First Prime"])
        if y_val >= 2 ** output_dim:
            y_val = y_val % (2 ** output_dim)
        y_true = int_to_binvec(y_val, output_dim)
        y_pred = model(x).squeeze().round()

        if torch.equal(y_pred, y_true):
            correct += 1
        total += 1

print(f"\n✅ Validation Accuracy: {correct / total * 100:.2f}%")

