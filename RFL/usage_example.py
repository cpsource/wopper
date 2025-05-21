Usage Example
# Dimensions
input_dim = 128
hidden_dim = 64
depth = 4

# Instantiate
model = RFLNet(input_dim, hidden_dim, depth)

# Dummy Input
dummy_input = torch.randn(10, input_dim)  # batch of 10
output = model(dummy_input)
print(output)

