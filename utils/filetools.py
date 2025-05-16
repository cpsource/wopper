# utils/filetools.py

import os
import json
from typing import List, Dict, Any

def ensure_dir(path: str):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


def read_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Read a JSONL (JSON Lines) file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line.strip()}\n{e}")
    return data


def write_jsonl(filepath: str, data: List[Dict[str, Any]]):
    """Write a list of dictionaries to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')


def read_text_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(filepath: str, text: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)


# -----------------------------------
# 🧪 Local test
# -----------------------------------
if __name__ == "__main__":
    test_path = "test_data.jsonl"
    test_data = [
        {"sentence": "The dog barked loudly.", "subject": "dog", "action": "barked", "destination": ""},
        {"sentence": "The mailman walked to the door.", "subject": "mailman", "action": "walked", "destination": "door"}
    ]

    write_jsonl(test_path, test_data)
    loaded = read_jsonl(test_path)
    print("Read back:", loaded)

    write_text_file("example.txt", "Hello, filetools!")
    print("Text read:", read_text_file("example.txt"))

