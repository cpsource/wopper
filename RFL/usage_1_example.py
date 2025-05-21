# Instantiate model and optimizer
model = RFLNet(128, 64, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for batch_x, batch_y in dataloader:
    optimizer.zero_grad()
    
    # Forward pass with resonance tracking
    preds, resonance_stack = model(batch_x, return_resonance=True)
    
    # Compute custom RFL loss
    loss = rfl_loss(preds.squeeze(), batch_y.float(), resonance_stack)
    
    # Backpropagation and optimizer step
    loss.backward()
    optimizer.step()

