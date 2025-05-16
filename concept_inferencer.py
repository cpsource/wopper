import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class ConceptInferencer(nn.Module):
    def __init__(self, hidden_dim=256):
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

    print("Subject logits shape:", subj.shape)
    print("Action logits shape:", act.shape)
    print("Destination logits shape:", dest.shape)

if __name__ == "__main__":
    main()

