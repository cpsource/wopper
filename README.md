# wopper
Play with AI

## Repository Overview

This repository contains a small prototype called **"wopper"**. It mixes a
simple PyTorch model, some utility scripts, and thin interfaces to ChatGPT and
Wikidata. At the top level are training and inference scripts, utilities, and
interface modules.

### Core Components

1. **Data Loading & Vocabulary**
   - `dataset_loader.py` defines a `ConceptDataset` that reads JSONL lines,
     tokenizes them with BERT, and builds a vocabulary as it loads each entry.
     Every line yields IDs for "subject," "action," and "destination."
   - `sample_training_data.jsonl` contains three example sentences with those
     fields.
   - `vocab_manager.py` implements a simple token↔ID mapping with methods
     `add_token`, `get_id`, `get_token`, and so on.

2. **Model**
   - `concept_inferencer.py` wraps a pretrained BERT model and attaches three
     linear "heads" for predicting subject, action, and destination IDs from an
     input sentence.

3. **Training**
   - `train_concept_model.py` loads the dataset, trains the model with
     early-stopping, and saves the resulting weights plus the vocabulary. It
     logs progress and plots a loss curve at the end.

4. **Inference**
   - `inference.py` demonstrates running the saved model on a sentence. It
     loads the vocabulary and model weights, predicts the concept IDs, and
     translates them back to tokens for display.

5. **Interfaces**
   - `interface/chatgpt_interface.py` provides a minimal wrapper around the
     OpenAI API, expecting an API key in `~/.env`.
   - `interface/wikidata_interface.py` offers a helper to run SPARQL queries
     against Wikidata; the example method `get_subclasses` fetches subclasses
     for a concept ID.

6. **Utilities and Extra Scripts**
   - `utils/` contains small helpers for reading/writing text and JSONL files.
   - `utils/auto_programmer.py` uses ChatGPT to iteratively generate code, test it,
     and save the result in `utils/` when it passes.
   - `rebuild_and_infer.sh` shows a workflow: delete old artifacts, train, then
     run inference.

### Documentation & Tests

- The `doc/` folder contains conversational notes about design ideas (SPARQL,
  concept inference, etc.).
- `test/wopper_test.py` currently exercises `WikidataInterface` and hints at
  future tests.

### Getting Started

1. Inspect or modify `sample_training_data.jsonl` to add more examples.
2. Run `train_concept_model.py` to train and save `concept_inferencer.pt` and
   `vocab.json`.
3. Use `inference.py` (or the shell script `rebuild_and_infer.sh`) to see
   predictions on new sentences.
4. Explore `interface/chatgpt_interface.py` and
   `interface/wikidata_interface.py` if you want to integrate ChatGPT prompts or
   SPARQL queries.
5. The `utils/auto_programmer.py` script demonstrates how ChatGPT can
   iteratively produce working Python utilities.
6. Run `pytest` to execute the smoke tests in `test/` which call each module's
   `main()` function when available.

### What to Learn Next

- **PyTorch** basics (loading data, defining modules, training loops).
- **Transformers/BERT** usage for sentence encoding.
- **SPARQL/Wikidata** queries if you plan to expand knowledge integration.
- How to manage a vocabulary and dataset for more substantial training data.
- The OpenAI API for automated code generation or interactive prompting.

Overall, the repository is a lightweight experiment connecting BERT-based
concept inference with simple data utilities and external interfaces. You can
extend it by adding more data, new concept fields, or richer SPARQL/ChatGPT
interactions.
