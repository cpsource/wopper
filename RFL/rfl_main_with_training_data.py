import torch
import torch.nn as nn
import torch.nn.functional as F

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

def rfl_loss(output, target, resonance_stack, alpha=1.0, beta=0.1):
    task_loss = F.binary_cross_entropy(output, target)
    resonance_loss = 0.0
    for i in range(1, len(resonance_stack)):
        diff = resonance_stack[i] - resonance_stack[i - 1]
        resonance_loss += torch.mean(torch.norm(diff, dim=1))
    resonance_loss = resonance_loss / (len(resonance_stack) - 1)
    return alpha * task_loss + beta * resonance_loss

def encode_text_to_vector(text, dim=128):
    torch.manual_seed(abs(hash(text)) % (2**32))  # simple deterministic encoding
    return torch.randn(dim)

def main():
    # Define training examples from categories 2, 3, and 5
    training_texts = [
        # Echo Reinforcement Dataset (Example 2)
        "Freedom is the absence of fear",
        "Unchained thought breathes easiest",
        "He walked without caution or chains",

        # Contradiction Decay Sequences (Example 3)
        "She believed in truth",
        "She questioned the facts",
        "She deleted the evidence",

        # Echo Chain Compression (Example 5)
        "Knowledge is stored in books",
        "Books are libraries compressed",
        "Memory is a library made of synapses"
    ]

    # Assign labels: first group 1.0 (resonant), second group 0.0 (decay), third group 1.0 (resonant)
    labels = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    # Convert text to feature vectors
    input_dim = 128
    inputs = torch.stack([encode_text_to_vector(t, input_dim) for t in training_texts])
    targets = torch.tensor(labels).unsqueeze(1)

    # Model setup
    hidden_dim = 64
    depth = 4
    model = RFLNet(input_dim, hidden_dim, depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop for a few epochs
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        for i in range(len(inputs)):
            x = inputs[i].unsqueeze(0)  # batch of 1
            y = targets[i]
            optimizer.zero_grad()
            preds, resonance_stack = model(x, return_resonance=True)
            loss = rfl_loss(preds, y.unsqueeze(0), resonance_stack)
            #loss = rfl_loss(preds.squeeze(), y, resonance_stack)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Final predictions
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(training_texts):
            x = inputs[i].unsqueeze(0)
            pred = model(x)
            print(f'"{text}" → Prediction: {pred.item():.4f} (Label: {labels[i]})')

if __name__ == "__main__":
    main()
