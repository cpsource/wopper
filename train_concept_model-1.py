import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from concept_inferencer import ConceptInferencer
from dataset_loader import ConceptDataset
from vocab_manager import VocabManager


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
    epochs = 1116
    batch_size = 4
    lr = 2e-5

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab and dataset
    vocab = VocabManager()
    dataset = ConceptDataset("sample_training_data.jsonl", vocab)
    vocab.freeze()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, optimizer, loss
    model = ConceptInferencer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, epochs + 1):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "concept_inferencer.pt")
    vocab.save("vocab.json")
    print("Model and vocab saved.")


if __name__ == "__main__":
    main()

