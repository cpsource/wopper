import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ==================== HYPERPARAMETERS ====================
# Data parameters
INPUT_DIM = 20
OUTPUT_DIM = 20
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model architecture parameters
HIDDEN_DIM = 100
NUM_HIDDEN_LAYERS = 10

# Training parameters
BATCH_SIZE = 25
LEARNING_RATE = 0.001
EPOCHS = 10

# Initialization parameters
BIAS_INIT_VALUE = 0.5

# File paths
DATA_FILE = "prime_factors_output.csv"
MODEL_SAVE_PATH = "deep_binary_net.pth"
# =========================================================

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Step 1: Define the initializer function
def custom_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)        # or kaiming_uniform_, normal_, etc.
        nn.init.constant_(m.bias, BIAS_INIT_VALUE)          # custom bias init
        
# Binary conversion utility
def int_to_binvec(n, width):
    return torch.tensor([int(b) for b in f"{n:0{width}b}"], dtype=torch.float32)

# Custom Dataset class
class PrimeDataset(Dataset):
    def __init__(self, dataframe, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM):
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
        return x.to(device), y.to(device)

# Neural Network with Batch Normalization
class DeepBinaryNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS):
        super().__init__()
        
        # First layer: Input -> Hidden (with BatchNorm)
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()  # Changed from Sigmoid to ReLU for better gradient flow
        ]
        
        # Hidden layers: Hidden -> Hidden (with BatchNorm)
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ]
        
        # Output layer: No BatchNorm here since we want specific output range
        layers += [
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Keep Sigmoid for binary output
        ]
        
        self.model = nn.Sequential(*layers)
        
        # Apply custom initializer only to Linear layers
        self.model.apply(custom_init)
    
    def forward(self, x):
        return self.model(x)

# Load dataset
data = pd.read_csv(DATA_FILE)
train_df, val_df = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Dataset and DataLoader
train_dataset = PrimeDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model setup
model = DeepBinaryNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Save the trained model's state dictionary
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for _, row in val_df.iterrows():
        x_val = int(row["Number"]) % (2 ** INPUT_DIM)
        y_val = int(row["First Prime"]) % (2 ** OUTPUT_DIM)
        x = int_to_binvec(x_val, INPUT_DIM).unsqueeze(0).to(device)
        y_true = int_to_binvec(y_val, OUTPUT_DIM).to(device)
        y_pred = model(x).squeeze().round()
        if torch.equal(y_pred, y_true):
            correct += 1
        total += 1

print(f"\n✅ Validation Accuracy: {correct / total * 100:.2f}%")

