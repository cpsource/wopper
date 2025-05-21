import unittest
import torch
from rfl_model import RFLNet

class TestRFLNet(unittest.TestCase):
    def test_forward_pass(self):
        input_dim = 128
        hidden_dim = 64
        depth = 4

        model = RFLNet(input_dim, hidden_dim, depth)
        dummy_input = torch.randn(10, input_dim)  # batch of 10
        output = model(dummy_input)

        # Check output shape
        self.assertEqual(output.shape, (10, 1))

        # Check output values are between 0 and 1 (sigmoid)
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

if __name__ == '__main__':
    unittest.main()