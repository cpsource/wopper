# dataset_loader.py
# Reads training examples from JSONL and tokenizes them with BERT.

import json
import os
from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils.logger import get_logger

log = get_logger(__name__)
log.debug("Starting dataset_loader.py")

class ConceptDataset(Dataset):
    def __init__(self, jsonl_file, vocab_manager):
        load_dotenv(os.path.expanduser("~/.env"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in ~/.env")
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

                    log.debug(
                        f"[Line {line_num}] {sentence} → subj: {subject_id} ({subject}), act: {action_id} ({action}), dest: {destination_id} ({destination})"
                    )

                    self.data.append({
                        'input_ids': tokenized['input_ids'].squeeze(0),
                        'attention_mask': tokenized['attention_mask'].squeeze(0),
                        'subject_id': subject_id,
                        'action_id': action_id,
                        'destination_id': destination_id
                    })
                except Exception as e:
                    log.error(f"Error parsing line {line_num}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

