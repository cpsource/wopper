import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Step 1: Define the initializer function
def custom_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)        # or kaiming_uniform_, normal_, etc.
        nn.init.constant_(m.bias, 0.5)          # custom bias init
        
# Binary conversion utility
def int_to_binvec(n, width):
    return torch.tensor([int(b) for b in f"{n:0{width}b}"], dtype=torch.float32)

# Custom Dataset class
class PrimeDataset(Dataset):
    def __init__(self, dataframe, input_dim=20, output_dim=20):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.samples = []
        for _, row in dataframe.iterrows():
            x_val = int(row["Number"]) % (2 ** input_dim)
            y_val = int(row["First Prime"]) % (2 ** output_dim)
            x = int_to_binvec(x_val, input_dim)
            y = int_to_binvec(y_val, output_dim)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        #print(f"x = {x}, y = {y}")
        return x.to(device), y.to(device)

# Neural Network
class DeepBinaryNet(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=100, output_dim=20, num_hidden_layers=10):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]
        self.model = nn.Sequential(*layers)
        # Step 3: Apply the custom initializer
        self.model.apply(custom_init)

    def forward(self, x):
        return self.model(x)

# Load dataset
data = pd.read_csv("prime_factors_output.csv")
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

# Dataset and DataLoader
batch_size = 25
train_dataset = PrimeDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model setup
input_dim = 20
output_dim = 20
model = DeepBinaryNet(input_dim=input_dim, output_dim=output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    # Save the trained model's state dictionary
    torch.save(model.state_dict(), "deep_binary_net.pth")
    print("✅ Model saved to deep_binary_net.pth")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for _, row in val_df.iterrows():
        x_val = int(row["Number"]) % (2 ** input_dim)
        y_val = int(row["First Prime"]) % (2 ** output_dim)
        x = int_to_binvec(x_val, input_dim).unsqueeze(0).to(device)
        y_true = int_to_binvec(y_val, output_dim).to(device)
        y_pred = model(x).squeeze().round()
        if torch.equal(y_pred, y_true):
            correct += 1
        total += 1

print(f"\n✅ Validation Accuracy: {correct / total * 100:.2f}%")

