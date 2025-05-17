# Neural model using BERT to predict subject, action and destination.
import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from utils.logger import get_logger

log = get_logger(__name__)
log.debug("Starting concept_inferencer.py")

class ConceptInferencer(nn.Module):
    def __init__(self, hidden_dim=256):
        load_dotenv(os.path.expanduser("~/.env"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in ~/.env")
        super(ConceptInferencer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()

        # Output heads for each concept field
        self.subject_head = nn.Linear(hidden_dim, 1000)       # 1000 = vocab limit
        self.action_head = nn.Linear(hidden_dim, 1000)
        self.destination_head = nn.Linear(hidden_dim, 1000)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.relu(self.hidden(outputs.pooler_output))
        
        subject_logits = self.subject_head(x)
        action_logits = self.action_head(x)
        destination_logits = self.destination_head(x)
        
        return subject_logits, action_logits, destination_logits

# --------------------------
# 🧪 Local test function
# --------------------------
def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = ConceptInferencer()
    model.eval()

    sentence = "The girl went to the grocery store."
    inputs = tokenizer(sentence, return_tensors="pt")
    subj, act, dest = model(inputs['input_ids'], inputs['attention_mask'])

    log.info("Subject logits shape: %s", subj.shape)
    log.info("Action logits shape: %s", act.shape)
    log.info("Destination logits shape: %s", dest.shape)

if __name__ == "__main__":
    main()

