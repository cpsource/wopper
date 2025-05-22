import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import BertTokenizer, BertModel

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
    def __init__(self, input_dim, hidden_dim, depth):
        super(RFLNet, self).__init__()
        self.blocks = nn.ModuleList([
            RFLBlock(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(depth)
        ])
        self.final = nn.Linear(hidden_dim, 1)

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
    """Encodes text with BERT and projects to ``output_dim`` for the RFL network."""

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.bert.config.hidden_size, output_dim)

    def encode(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            vec = self.bert(**tokens).pooler_output
        return self.proj(vec).squeeze(0)

    def forward(self, texts):
        tokens = self.tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            vec = self.bert(**tokens).pooler_output
        return self.proj(vec)


def main():
    # Test examples (Echo Reinforcement)
    test_texts = [
        "Freedom is the absence of fear",
        "Unchained thought breathes easiest",
        "He walked without caution or chains"
    ]

    input_dim = 128
    hidden_dim = 64
    depth = 4
    model_path = "rfl_trained.pt"

    frontend = BERTFrontend(input_dim)

    # Instantiate and load pretrained model
    model = RFLNet(input_dim, hidden_dim, depth)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded pretrained model from {model_path}")
    else:
        raise FileNotFoundError(
            f"Pretrained model not found at {model_path}. Please train the model first."
        )

    model.eval()

    # Predict
    with torch.no_grad():
        for text in test_texts:
            x = frontend.encode(text).unsqueeze(0)
            pred = model(x)
            print(f'"{text}" → Prediction: {pred.item():.4f}')

if __name__ == "__main__":
    main()