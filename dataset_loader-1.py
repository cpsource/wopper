# Loads JSONL concept data and builds a dataset for training.
import json
import os
from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from vocab_manager import VocabManager
from logger import get_logger

log = get_logger(__name__)
log.debug("Starting dataset_loader-1.py")

class ConceptDataset(Dataset):
    def __init__(self, data_path, vocab_manager):
        load_dotenv(os.path.expanduser("~/.env"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in ~/.env")
        self.data = []
        self.vocab = vocab_manager
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        with open(data_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue # skip empty lines
                example = json.loads(line)
                sentence = example["sentence"]
                subject = example["subject"]
                action = example["action"]
                destination = example["destination"]

                self.vocab.add_word(subject)
                self.vocab.add_word(action)
                self.vocab.add_word(destination)

                self.data.append({
                    "sentence": sentence,
                    "subject_id": self.vocab.get_id(subject),
                    "action_id": self.vocab.get_id(action),
                    "destination_id": self.vocab.get_id(destination)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        inputs = self.tokenizer(example["sentence"], return_tensors="pt", padding="max_length", truncation=True, max_length=32)

        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "subject_id": torch.tensor(example['subject_id'], dtype=torch.long),
            "action_id": torch.tensor(example['action_id'], dtype=torch.long),
            "destination_id": torch.tensor(example['destination_id'], dtype=torch.long)
        }

# --------------------------
# 🧪 Local test function
# --------------------------
def main():
    vm = VocabManager()
    dataset = ConceptDataset("sample_training_data.jsonl", vm)
    vm.freeze()

    log.info(f"Size of dataset: {len(dataset)}")
    sample = dataset[0]
    log.info("Sample: %s", {k: v.shape if hasattr(v, 'shape') else v for k, v in sample.items()})

if __name__ == "__main__":
    main()

