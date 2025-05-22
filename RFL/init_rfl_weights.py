import torch.nn as nn
import torch.nn.init as init

def init_rfl_weights(module):
    """
    Custom weight initializer for RFLNet layers.
    - Uses Xavier initialization for Linear weights
    - Constant initialization for biases
    """
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)       # good for tanh/sigmoid
        init.constant_(module.bias, 0.01)         # small positive bias

# Usage:
# model = RFLNet(input_dim, hidden_dim, depth)
# model.apply(init_rfl_weights)

#
# used as follows
#

"""
from init_rfl_weights import init_rfl_weights

model = RFLNet(input_dim, hidden_dim, depth)
model.apply(init_rfl_weights)
"""

