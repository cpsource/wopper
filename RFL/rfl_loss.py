import torch
import torch.nn.functional as F

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