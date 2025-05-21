import torch

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