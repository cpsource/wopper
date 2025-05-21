# Reflexive Fractal Logic Network in PyTorch

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

    def forward(self, x):
        resonance = None
        for block in self.blocks:
            x = block(x, resonance)
            resonance = x
        return torch.sigmoid(self.final(x))

