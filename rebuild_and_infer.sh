#!/bin/bash

# Rebuild and rerun inference from scratch after adding new data

set -e  # Exit immediately if a command exits with a non-zero status
cd "$(dirname "$0")"  # Ensure we're in the wopper directory

echo "🧹 Cleaning up old model artifacts..."
rm -f concept_inferencer.pt vocab.json

echo "🧠 Re-training the concept inference model..."
python3 train_concept_model.py

echo "🤖 Running inference..."
python3 inference.py

