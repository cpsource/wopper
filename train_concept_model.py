# train_concept_model.py
# Trains the ConceptInferencer on the sample dataset with early stopping.

from logger import get_logger
log = get_logger(__name__)

log.info("Starting train_concept_model")

# suggestions
#log.info(f"Generated program saved to: {output_path}")
#log.debug(f"Response content:\n{response[:500]}")
#log.warning(f"ChatGPT retry due to error: {stderr}")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from concept_inferencer import ConceptInferencer
from dataset_loader import ConceptDataset
from vocab_manager import VocabManager
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        subject_id = batch['subject_id'].to(device)
        action_id = batch['action_id'].to(device)
        destination_id = batch['destination_id'].to(device)

        optimizer.zero_grad()
        subj_logits, act_logits, dest_logits = model(input_ids, attention_mask)

        loss = criterion(subj_logits, subject_id) + \
               criterion(act_logits, action_id) + \
               criterion(dest_logits, destination_id)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Hyperparameters
    max_epochs = 1000
    batch_size = 4
    lr = 2e-5
    patience = 50

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab and dataset
    vocab = VocabManager()
    dataset = ConceptDataset("sample_training_data.jsonl", vocab)
    vocab.freeze()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, optimizer, loss
    model = ConceptInferencer().to(device)

    log.debug(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training with early stopping
    best_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    loss_history = []

    for epoch in range(1, max_epochs + 1):
        loss = train(model, dataloader, optimizer, criterion, device)
        loss_history.append(loss)
        log.info(f"Epoch {epoch}/{max_epochs} - Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            log.info(f"Early stopping after {epoch} epochs.")
            break

    # Save the best model
    if best_model:
        model.load_state_dict(best_model)
        torch.save(model.state_dict(), "concept_inferencer.pt")
        vocab.save("vocab.json")
        log.info("Model and vocab saved.")

    # Plot loss curve
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

