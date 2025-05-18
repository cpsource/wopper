import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # Suppress INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # Optional: turn off oneDNN to suppress related messages

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import numpy as np

def softmax(x, temperature=1.0):
    x = np.array(x)
    x = x / temperature
    e_x = np.exp(x - np.max(x))  # numerical stability
    return e_x / e_x.sum()

class DriveScorer:
    DRIVES = [
        "Survival", "Sleep", "Reproduction", "Belonging", "Curiosity",
        "Control", "Status", "Joy", "Avoid Pain", "Nurture"
    ]

    def __init__(self, seed_data_path="drive_seed_data.json", log_path="drive_scorer_log.json"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.seed_data_path = seed_data_path
        self.log_path = log_path
        self.seed_sentences = []
        self.seed_vectors = []
        self.seed_scores = []
        self.load_seed_data()

    def load_seed_data(self):
        if os.path.exists(self.seed_data_path):
            with open(self.seed_data_path, "r") as f:
                raw_data = json.load(f)
            for item in raw_data:
                self.seed_sentences.append(item["sentence"])
                self.seed_vectors.append(self.model.encode(item["sentence"]))

                #print(item)
                #print(self.model.encode(item["sentence"]))
                
                self.seed_scores.append(np.array(item["scores"]))
        else:
            self._initialize_seed_data()

    def _initialize_seed_data(self):
        seed_data = [
            {"sentence": "I ate lunch", "scores": [0.85, 0.05, 0.0, 0.2, 0.05, 0.1, 0.02, 0.15, 0.4, 0.03]},
            {"sentence": "I need a nap", "scores": [0.2, 0.95, 0, 0.1, 0, 0, 0, 0.1, 0.5, 0.1]},
            {"sentence": "I want to be the best in my field", "scores": [0.1, 0, 0.1, 0.2, 0.2, 0.5, 0.95, 0.1, 0.1, 0.1]},
            {"sentence": "I just ran a marathon.", "scores": [0.088, 0.044, 0.022, 0.132, 0.066, 0.154, 0.176, 0.110, 0.044, 0.022]}
        ]
        with open(self.seed_data_path, "w") as f:
            json.dump(seed_data, f, indent=2)
        self.load_seed_data()

    def score(self, sentence, top_k=2):
        vec = self.model.encode(sentence)
        if not self.seed_vectors:
            return [0.0] * len(self.DRIVES)

        sims = cosine_similarity([vec], self.seed_vectors)[0]

        print(f"sims = {sims}")
        
        top_idxs = sims.argsort()[-top_k:][::-1]

        print(f"top_idxs = {top_idxs}")

        # Set your sharpness here
        temperature = 0.3

        # Select top matches
        weights = softmax([sims[i] for i in top_idxs], temperature=temperature)

        print(f"weights = {weights}")

        weighted_scores = np.zeros(len(self.DRIVES))
        for idx, i in enumerate(top_idxs):
            weighted_scores += self.seed_scores[i] * weights[idx]

        result = weighted_scores.tolist()

        if max(sims) < 0.5:
            self.log_low_confidence(sentence, result, max(sims))

        return result

    def learn(self, sentence, scores):
        new_entry = {"sentence": sentence, "scores": scores}
        if os.path.exists(self.seed_data_path):
            with open(self.seed_data_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(new_entry)
        with open(self.seed_data_path, "w") as f:
            json.dump(data, f, indent=2)

        self.seed_sentences.append(sentence)
        self.seed_vectors.append(self.model.encode(sentence))
        self.seed_scores.append(np.array(scores))

    def log_low_confidence(self, sentence, scores, similarity):
        log_entry = {
            "sentence": sentence,
            "scores": [float(x) for x in scores],
            "max_similarity": float(similarity)
        }

        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []
        else:
            log_data = []
    
        log_data.append(log_entry)
        with open(self.log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        
    def explain(self):
        return self.DRIVES


def run_tasklet(**kwargs):
    sentence = kwargs.get("sentence")
    if not sentence:
        raise ValueError("Tasklet requires a 'sentence' argument.")

    scorer = DriveScorer()
    scores = scorer.score(sentence)
    return dict(zip(scorer.explain(), scores))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python drive_scorer.py \"Your sentence here\"")
        sys.exit(1)

    scorer = DriveScorer()
    input_sentence = sys.argv[1]
    results = scorer.score(input_sentence)

    print(f"Sentence: {input_sentence}\n")
    for drive, score in zip(scorer.explain(), results):
        print(f"{drive:15}: {score:.2f}")

