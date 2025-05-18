import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'           # Suppress INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # Optional: turn off oneDNN to suppress related messages

from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import faiss

class DriveScorer:
    DRIVES = [
        "Survival", "Sleep", "Reproduction", "Belonging", "Curiosity",
        "Control", "Status", "Joy", "Avoid Pain", "Nurture"
    ]

    def __init__(self, seed_data_path="drive_seed_data.json", log_path="drive_scorer_log.json"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.seed_data_path = seed_data_path
        self.index_path = "drive_index.faiss"
        self.log_path = log_path
        self.seed_sentences = []
        self.seed_vectors = []
        self.seed_scores = []
        self.index = None  # FAISS index
        self.load_seed_data()

    def load_seed_data(self):

        if os.path.exists(self.seed_data_path):
            with open(self.seed_data_path, "r") as f:
                raw_data = json.load(f)

            self.seed_sentences = []
            self.seed_vectors = []
            self.seed_scores = []

            for item in raw_data:
                self.seed_sentences.append(item["sentence"])
                vec = self.model.encode(item["sentence"])
                self.seed_vectors.append(vec)
                self.seed_scores.append(np.array(item["scores"]))

            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            else:
                self._build_faiss_index()
                faiss.write_index(self.index, self.index_path)
        else:
            self._initialize_seed_data()
        
    def _initialize_seed_data(self):
        seed_data = [
            {"sentence": "I ate lunch", "scores": [0.85, 0.05, 0.0, 0.2, 0.05, 0.1, 0.02, 0.15, 0.4, 0.03]},
            {"sentence": "I need a nap", "scores": [0.2, 0.95, 0, 0.1, 0, 0, 0, 0.1, 0.5, 0.1]},
            {"sentence": "I want to be the best in my field", "scores": [0.1, 0, 0.1, 0.2, 0.2, 0.5, 0.95, 0.1, 0.1, 0.1]},
            {"sentence": "I just ran a marathon", "scores": [0.088, 0.044, 0.022, 0.132, 0.066, 0.154, 0.176, 0.110, 0.044, 0.022]}
        ]
        with open(self.seed_data_path, "w") as f:
            json.dump(seed_data, f, indent=2)
        self.load_seed_data()

    def _build_faiss_index(self):
        dimension = len(self.seed_vectors[0])
        self.index = faiss.IndexFlatIP(dimension)
        matrix = np.array(self.seed_vectors).astype("float32")
        faiss.normalize_L2(matrix)
        self.index.add(matrix)
        self.normalized_seed_vectors = matrix  # Save for re-normalizing queries
        faiss.write_index(self.index, self.index_path)

    def softmax(self, x, temperature=0.3):
        x = np.array(x) / temperature
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def score(self, sentence, top_k=3):
        if not self.index:
            return [0.0] * len(self.DRIVES)

        vec = self.model.encode(sentence)
        vec = np.array([vec]).astype("float32")
        faiss.normalize_L2(vec)

        sims, top_idxs = self.index.search(vec, top_k)
        sims = sims[0]
        top_idxs = top_idxs[0]

        weights = self.softmax(sims, temperature=0.1)
        weighted_scores = np.zeros(len(self.DRIVES))
        for idx, i in enumerate(top_idxs):
            weighted_scores += self.seed_scores[i] * weights[idx]

        result = weighted_scores.tolist()

        if max(sims) < 0.5:
            self.log_low_confidence(sentence, result, float(max(sims)))

        return result

    def learn(self, sentence, scores):
        vec = self.model.encode(sentence)
        self.seed_sentences.append(sentence)
        self.seed_vectors.append(vec)
        self.seed_scores.append(np.array(scores))

        vec_np = np.array([vec]).astype("float32")
        faiss.normalize_L2(vec_np)

        if self.index is None:
            self._build_faiss_index()
        else:
            self.index.add(vec_np)

        updated_data = []
        for s, score_array in zip(self.seed_sentences, self.seed_scores):
            updated_data.append({
                "sentence": s,
                "scores": [float(x) for x in score_array]
            })

        with open(self.seed_data_path, "w") as f:
            json.dump(updated_data, f, indent=2)

        faiss.write_index(self.index, self.index_path)
    
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

