# vocab_manager.py

import json
import os
from dotenv import load_dotenv
from logger import get_logger

log = get_logger(__name__)
log.debug("Starting vocab_manager.py")

class VocabManager:
    def __init__(self):
        load_dotenv(os.path.expanduser("~/.env"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in ~/.env")
        self.token_to_id = {}
        self.id_to_token = {}
        self.next_id = 0
        self._frozen = False

    def add_token(self, token):
        if self._frozen:
            return
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1

    def get_id(self, token):
        return self.token_to_id.get(token, self.token_to_id.get("<UNK>", -1))

    def get_token(self, idx):
        return self.id_to_token.get(idx, "<UNK>")

    def freeze(self):
        self._frozen = True
        if "<UNK>" not in self.token_to_id:
            self.add_token("<UNK>")

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({"token_to_id": self.token_to_id}, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.token_to_id = data["token_to_id"]
            self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
            self.next_id = max(self.id_to_token.keys()) + 1
            self._frozen = True

if __name__ == "__main__":
    vm = VocabManager()
    for word in ["cat", "sat", "mat"]:
        vm.add_token(word)
    vm.freeze()

    log.info("Token to ID: %s", vm.token_to_id)
    log.info("ID to Token: %s", vm.id_to_token)
    log.info("ID for 'sat': %s", vm.get_id("sat"))
    log.info("Token for ID 2: %s", vm.get_token(2))

    vm.save("test_vocab.json")
    log.info("Saved vocab to test_vocab.json")
