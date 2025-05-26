import torch
import torch.nn as nn
import pandas as pd

# ==================== HYPERPARAMETERS ====================
# Data parameters
INPUT_DIM = 20
OUTPUT_DIM = 20

# Model architecture parameters
HIDDEN_DIM = 100
NUM_HIDDEN_LAYERS = 10

# File paths
MODEL_SAVE_PATH = "deep_binary_net.pth"
TEST_DATA_FILE = "test_factors_output.csv"

# Configuration
TRACE_FLAG = False  # Set to True for detailed tracing (recommended for small samples)
SHOW_DETAILED_FIRST_N = 3  # Show detailed tracing for first N samples
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Format helper: round and convert tensor to list of floats with 2 decimal places
def format_tensor(t):
    return [round(float(v), 2) for v in t.squeeze()]

# Define the model architecture (must match training model)
class DeepBinaryNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS):
        super().__init__()
        
        # First layer: Input -> Hidden (with BatchNorm)
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
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
            nn.Sigmoid()
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        # Enable tracing only for first few samples or if globally enabled
        current_trace = TRACE_FLAG or (hasattr(self, '_current_sample') and self._current_sample < SHOW_DETAILED_FIRST_N)
        
        if current_trace:
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
                    
                    print(f"  Linear Output (Wx + b): {format_tensor(z)}")
                    x = z
                
                elif isinstance(layer, nn.BatchNorm1d):
                    # Apply batch normalization
                    x = layer(x)
                    print(f"Layer {i} (BatchNorm1d):")
                    print(f"  BatchNorm Output: {format_tensor(x)}")
                    print(f"  Running Mean: {format_tensor(layer.running_mean)}")
                    print(f"  Running Var: {format_tensor(layer.running_var)}")
                
                elif isinstance(layer, nn.ReLU):
                    x = torch.relu(x)
                    print(f"Layer {i} (ReLU):")
                    print(f"  ReLU Output: {format_tensor(x)}")
                
                elif isinstance(layer, nn.Sigmoid):
                    x = torch.sigmoid(x)
                    print(f"Layer {i} (Sigmoid):")
                    print(f"  Sigmoid Output: {format_tensor(x)}")
                
                elif isinstance(layer, nn.Tanh):
                    x = torch.tanh(x)
                    print(f"Layer {i} (Tanh):")
                    print(f"  Tanh Output: {format_tensor(x)}")
            
            print("\n✅ Final Output:", format_tensor(x))
            return x
        else:
            return self.model(x)

# Convert integer to fixed-width binary vector
def int_to_binvec(n, width):
    return torch.tensor([int(b) for b in f"{n:0{width}b}"], dtype=torch.float32)

# Load model and weights
model = DeepBinaryNet().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

print(f"✅ Model loaded from {MODEL_SAVE_PATH}")
print(f"✅ Model architecture: {INPUT_DIM} -> {HIDDEN_DIM} (x{NUM_HIDDEN_LAYERS} layers) -> {OUTPUT_DIM}")
print(f"✅ Using device: {device}")

# Load test data
df = pd.read_csv(TEST_DATA_FILE)
print(f"✅ Test data loaded from {TEST_DATA_FILE}")

# Evaluate with tracing
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for i, row in df.iterrows():
        # Set current sample index for selective tracing
        model._current_sample = i
        
        # Show detailed output for first few samples
        if i < SHOW_DETAILED_FIRST_N:
            print(f"\n{'='*50}")
            print(f"=== Example {i+1}: Number = {row['Number']} ===")
            print(f"{'='*50}")
        
        x_val = int(row["Number"]) % (2 ** INPUT_DIM)
        x = int_to_binvec(x_val, INPUT_DIM).unsqueeze(0).to(device)
        
        if i < SHOW_DETAILED_FIRST_N:
            print(f"📊 Input Number: {row['Number']}")
            print(f"📊 Modded Input: {x_val}")
            print(f"📊 Binary Input: {format_tensor(x)}")
        
        # Run inference
        output = model(x)
        
        # Show expected output if available
        if "First Prime" in row:
            y_val = int(row["First Prime"]) % (2 ** OUTPUT_DIM)
            y_true = int_to_binvec(y_val, OUTPUT_DIM).to(device)  # Move to same device
            
            if i < SHOW_DETAILED_FIRST_N:
                print(f"\n📌 Expected First Prime: {row['First Prime']}")
                print(f"📌 Modded Expected: {y_val}")
                print(f"📌 Binary Expected: {format_tensor(y_true)}")
                print(f"🎯 Rounded Prediction: {format_tensor(output.round())}")
            
            # Check if prediction matches (ensure both tensors are on same device)
            is_correct = torch.equal(output.round().squeeze(), y_true)
            if is_correct:
                correct_predictions += 1
                if i < SHOW_DETAILED_FIRST_N:
                    print("✅ CORRECT PREDICTION!")
                elif i < 20:  # Show status for first 20 samples
                    print(f"Sample {i+1}: ✅ CORRECT (Number: {row['Number']}, Prime: {row['First Prime']})")
            else:
                if i < SHOW_DETAILED_FIRST_N:
                    print("❌ INCORRECT PREDICTION")
                elif i < 20:  # Show status for first 20 samples
                    print(f"Sample {i+1}: ❌ INCORRECT (Number: {row['Number']}, Prime: {row['First Prime']})")
            
            total_predictions += 1
        
        # Show progress for remaining samples
        if i >= 20 and (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(df)} samples... Current accuracy: {correct_predictions/total_predictions*100:.1f}%")

print(f"\n{'='*50}")
print("🏁 FINAL RESULTS")
print(f"{'='*50}")
print(f"📊 Total samples tested: {total_predictions}")
print(f"✅ Correct predictions: {correct_predictions}")
print(f"❌ Incorrect predictions: {total_predictions - correct_predictions}")
print(f"🎯 Overall accuracy: {correct_predictions/total_predictions*100:.2f}%")
print(f"{'='*50}")

