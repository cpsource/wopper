import torch
import torch.nn as nn
import torch.nn.functional as F


class RFLBlock(nn.Module):
    """
    Reflexive Fractal Logic Block:
    - Accepts input + optional past resonance
    - Projects input
    - Blends current and prior representations using a learned gate
    """
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
    """
    Reflexive Fractal Logic Network:
    - Stack of RFLBlocks
    - Echoing resonance pattern
    """
    def __init__(self, input_dim, hidden_dim, depth):
        super(RFLNet, self).__init__()
        self.blocks = nn.ModuleList([
            RFLBlock(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(depth)
        ])
        self.final = nn.Linear(hidden_dim, 1)  # For binary classification or scoring

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
    """
    output: final model prediction
    target: true labels
    resonance_stack: list of intermediate layer outputs (from RFLNet)
    alpha: weight for task loss
    beta: weight for internal resonance loss
    """
    # Task loss (binary classification example)
    task_loss = F.binary_cross_entropy(output, target)

    # Resonance loss: encourage internal layers to converge across time
    resonance_loss = 0.0
    for i in range(1, len(resonance_stack)):
        diff = resonance_stack[i] - resonance_stack[i - 1]
        resonance_loss += torch.mean(torch.norm(diff, dim=1))  # L2 norm between layers

    resonance_loss = resonance_loss / (len(resonance_stack) - 1)

    # Total loss
    return alpha * task_loss + beta * resonance_loss


def main():
    # Example setup for testing
    input_dim = 128
    hidden_dim = 64
    depth = 4
    batch_size = 10

    # Dummy dataset
    dummy_input = torch.randn(batch_size, input_dim)
    dummy_target = torch.randint(0, 2, (batch_size,)).float()

    # Instantiate model and optimizer
    model = RFLNet(input_dim, hidden_dim, depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simulate one training step
    model.train()
    optimizer.zero_grad()
    preds, resonance_stack = model(dummy_input, return_resonance=True)
    loss = rfl_loss(preds.squeeze(), dummy_target, resonance_stack)
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
    print("Output shape:", preds.shape)
    print("Output:", preds.detach().numpy())


if __name__ == "__main__":
    main()