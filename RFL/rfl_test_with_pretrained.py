import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class RFLBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RFLBlock, self).__init__()
        self.proj_input = nn.Linear(input_dim, hidden_dim)
        self.proj_resonance = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, past_resonance=None):
        h = torch.tanh(self.proj_input(x))

        if past_resonance is not None:
            echo = torch.tanh(self.proj_resonance(past_resonance))
            concat = torch.cat([h, echo], dim=-1)
            weight = torch.sigmoid(self.gate(concat))
            h = h * weight + echo * (1 - weight)

        return h

class RFLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth):
        super(RFLNet, self).__init__()
        self.blocks = nn.ModuleList([
            RFLBlock(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(depth)
        ])
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_resonance=False):
        resonance = None
        resonance_stack = []
        for block in self.blocks:
            x = block(x, resonance)
            resonance = x
            if return_resonance:
                resonance_stack.append(x)
        out = torch.sigmoid(self.final(x))
        return (out, resonance_stack) if return_resonance else out

def encode_text_to_vector(text, dim=128):
    torch.manual_seed(abs(hash(text)) % (2**32))  # Simple deterministic encoding
    return torch.randn(dim)

def main():
    # Test examples (Echo Reinforcement)
    test_texts = [
        "Freedom is the absence of fear",
        "Unchained thought breathes easiest",
        "He walked without caution or chains"
    ]

    input_dim = 128
    hidden_dim = 64
    depth = 4
    model_path = "rfl_trained.pt"

    # Instantiate and load pretrained model
    model = RFLNet(input_dim, hidden_dim, depth)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded pretrained model from {model_path}")
    else:
        print(f"Pretrained model not found at {model_path}. Using untrained model.")

    model.eval()

    # Predict
    with torch.no_grad():
        for text in test_texts:
            x = encode_text_to_vector(text, input_dim).unsqueeze(0)
            pred = model(x)
            print(f'"{text}" → Prediction: {pred.item():.4f}')

if __name__ == "__main__":
    main()