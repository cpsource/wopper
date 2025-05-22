import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from difflib import SequenceMatcher


class RFLBlock(nn.Module):
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
    def __init__(self, input_dim, hidden_dim, depth, num_outputs: int = 1):
        super(RFLNet, self).__init__()
        self.blocks = nn.ModuleList(
            [
                RFLBlock(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(depth)
            ]
        )
        self.final = nn.Linear(hidden_dim, num_outputs)

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


class BERTFrontend(nn.Module):
    """Encodes text using a pretrained BERT model and projects to ``output_dim``."""

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.bert.config.hidden_size, output_dim)

    def encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            vec = self.bert(**tokens).pooler_output
        return self.proj(vec).squeeze(0)

    def forward(self, texts):
        tokens = self.tokenizer(
            list(texts), return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            vec = self.bert(**tokens).pooler_output
        return self.proj(vec)


def rfl_loss(output, target, resonance_stack, alpha=1.0, beta=0.1):
    task_loss = F.binary_cross_entropy(output, target)
    resonance_loss = 0.0
    for i in range(1, len(resonance_stack)):
        diff = resonance_stack[i] - resonance_stack[i - 1]
        resonance_loss += torch.mean(torch.norm(diff, dim=1))
    resonance_loss = resonance_loss / (len(resonance_stack) - 1)
    return alpha * task_loss + beta * resonance_loss


def main():
    # Define training examples from categories 2, 3, and 5
    training_texts = [
        "Freedom is the absence of fear",
        "Unchained thought breathes easiest",
        "He walked without caution or chains",
        "She believed in truth",
        "She questioned the facts",
        "She deleted the evidence",
        "Knowledge is stored in books",
        "Books are libraries compressed",
        "Memory is a library made of synapses",
    ]

    # Concept seeds used for the RFL models
    concepts = [
        "freedom",
        "fear",
        "truth",
        "knowledge",
        "emotion",
        "constraint",
        "love",
        "loss",
        "ambiguity",
        "intuition",
    ]

    # Generate similarity-based labels for each concept
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    labels = [[_similarity(text, c) for c in concepts] for text in training_texts]

    # Convert text to feature vectors using BERT
    input_dim = 128
    frontend = BERTFrontend(input_dim)
    # Precompute the BERT features without tracking gradients. Otherwise the
    # ``inputs`` tensor retains a computation graph which causes an error when
    # reused across epochs.
    with torch.no_grad():
        inputs = frontend(training_texts)
    targets = torch.tensor(labels)

    # Model setup
    hidden_dim = 64
    depth = 4
    model = RFLNet(input_dim, hidden_dim, depth, len(concepts))

    from init_rfl_weights import init_rfl_weights

    model.apply(init_rfl_weights)  # 👈 apply custom weight initialization here

    from seed_rfl_with_concepts import seed_rfl_with_concepts

    # precondition model for our concepts
    # 👇 Seed the first layer with your concept priors
    # ``seed_rfl_with_concepts`` expects an encoding function that optionally
    # takes a dimension argument. Wrap ``frontend.encode`` so it matches that
    # signature without altering the original method.
    seed_rfl_with_concepts(model, concepts, lambda text, *_: frontend.encode(text))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(35):
        total_loss = 0.0
        for i in range(len(inputs)):
            x = inputs[i].unsqueeze(0)
            y = targets[i]
            optimizer.zero_grad()
            preds, resonance_stack = model(x, return_resonance=True)
            loss = rfl_loss(preds, y.unsqueeze(0), resonance_stack)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "rfl_trained.pt")
    print("Model saved to rfl_trained.pt")

    # Final predictions
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(training_texts):
            x = inputs[i].unsqueeze(0)
            pred = model(x)
            print(
                f'"{text}" → Prediction: {pred.squeeze(0).tolist()} '
                f"(Label: {labels[i]})"
            )


if __name__ == "__main__":
    main()
