# Runs the trained model to predict concepts from a sentence.
import os
from dotenv import load_dotenv
import torch
from transformers import BertTokenizer
from concept_inferencer import ConceptInferencer
from vocab_manager import VocabManager
from utils.logger import get_logger

log = get_logger(__name__)
log.debug("Starting inference.py")


def predict_concepts(sentence, model, tokenizer, vocab, device):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=32).to(device)
    with torch.no_grad():
        subj_logits, act_logits, dest_logits = model(inputs['input_ids'], inputs['attention_mask'])

    subj_id = torch.argmax(subj_logits, dim=1).item()
    act_id = torch.argmax(act_logits, dim=1).item()
    dest_id = torch.argmax(dest_logits, dim=1).item()

    subject = vocab.get_token(subj_id)
    action = vocab.get_token(act_id)
    destination = vocab.get_token(dest_id)

    return subject, action, destination


def main():
    sentence = "The girl went to the grocery store."

    # Load vocab
    vocab = VocabManager()
    vocab.load("vocab.json")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = ConceptInferencer()
    model.load_state_dict(torch.load("concept_inferencer.pt", map_location=torch.device("cpu")))
    model.to("cpu")

    # Predict
    subject, action, destination = predict_concepts(sentence, model, tokenizer, vocab, "cpu")
    log.info(f"Sentence: {sentence}")
    log.info(f"Subject: {subject}\nAction: {action}\nDestination: {destination}")


if __name__ == "__main__":
    main()

