# dataset_loader.py

import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class ConceptDataset(Dataset):
    def __init__(self, jsonl_file, vocab_manager):
        self.data = []
        self.vocab = vocab_manager
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        with open(jsonl_file, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    example = json.loads(line)
                    sentence = example['sentence']
                    subject = example['subject']
                    action = example['action']
                    destination = example['destination']

                    self.vocab.add_token(subject)
                    self.vocab.add_token(action)
                    self.vocab.add_token(destination)

                    tokenized = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=32, return_tensors="pt")

                    subject_id = self.vocab.get_id(subject)
                    action_id = self.vocab.get_id(action)
                    destination_id = self.vocab.get_id(destination)

                    # Debug print
                    print(f"[Line {line_num}] {sentence} → subj: {subject_id} ({subject}), act: {action_id} ({action}), dest: {destination_id} ({destination})")

                    self.data.append({
                        'input_ids': tokenized['input_ids'].squeeze(0),
                        'attention_mask': tokenized['attention_mask'].squeeze(0),
                        'subject_id': subject_id,
                        'action_id': action_id,
                        'destination_id': destination_id
                    })
                except Exception as e:
                    print(f"Error parsing line {line_num}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

