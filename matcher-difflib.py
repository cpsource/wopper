# Simple lexical similarity scoring using difflib.SequenceMatcher
# Reuses training_texts and concept list from the RFL example scripts.

from difflib import SequenceMatcher


def similarity(a: str, b: str) -> float:
    """Return lexical similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main() -> None:
    # Training examples (same as in rfl_main_with_training_data.py)
    training_texts = [
        "Freedom is the absence of fear",
        "Unchained thought breathes easiest",
        "He walked without caution or chains",
        "She believed in truth",
        "She questioned the facts",
        "She deleted the evidence",
        "Knowledge is stored in books",
        "Books are libraries compressed",
        "Memory is a library made of synapses",
    ]

    # Concept seeds used for the RFL models
    concepts = [
        "freedom",
        "fear",
        "truth",
        "knowledge",
        "emotion",
        "constraint",
        "love",
        "loss",
        "ambiguity",
        "intuition",
    ]

    ranked = []
    for text in training_texts:
        scores = [(c, similarity(text, c)) for c in concepts]
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked.append([c for c, _ in scores])

    print(ranked)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
