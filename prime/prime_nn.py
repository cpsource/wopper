import csv
from pathlib import Path
import sys

# Ensure the repo root is on the path when executed directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logger import get_logger

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None
    nn = None


log = get_logger(__name__)


if torch is not None:
    class PrimeNet(nn.Module):
        """Simple feed-forward network with 10 hidden layers of size 100."""

        def __init__(self, in_size: int = 20, hidden_size: int = 100, out_size: int = 10):
            super().__init__()
            layers = []
            for i in range(10):
                input_dim = in_size if i == 0 else hidden_size
                layers.append(nn.Linear(input_dim, hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, out_size))
            layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)
else:  # pragma: no cover - fallback when torch unavailable
    class PrimeNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to use PrimeNet")


def _to_binary_tensor(number: int, bits: int):
    """Convert an integer to a float tensor of binary digits."""
    binary = [int(b) for b in format(number, f"0{bits}b")]
    return torch.tensor(binary, dtype=torch.float32)


def load_dataset(csv_path: Path):
    """Load CSV and convert columns to binary tensors."""
    inputs = []
    outputs = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = _to_binary_tensor(int(row["Number"]), 20)
            y = _to_binary_tensor(int(row["First Prime"]), 10)
            inputs.append(x)
            outputs.append(y)
    x_tensor = torch.stack(inputs)
    y_tensor = torch.stack(outputs)
    return TensorDataset(x_tensor, y_tensor)


def main():
    dataset_file = Path(__file__).with_name("prime_factors_output.csv")
    if torch is None:
        log.warning("PyTorch not available; skipping prime NN demo.")
        log.warning("Would have loaded: %s", dataset_file)
        return

    dataset = load_dataset(dataset_file)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = PrimeNet()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 2
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
        log.info("Epoch %d complete. Last batch loss %.4f", epoch + 1, loss.item())

    # Freeze the model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            preds = model(x_batch)
            val_loss += criterion(preds, y_batch).item() * len(x_batch)
    val_loss /= len(val_ds)
    log.info("Validation loss: %.4f", val_loss)


if __name__ == "__main__":
    main()
