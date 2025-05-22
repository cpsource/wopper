"""Simple training text dataset used for RFL examples.

This module defines ``training_texts`` as a list of 10,000 short sentences. A
portion mention the concept seed words used by ``rfl_train_with_training_and_save``;
the rest cover unrelated topics with basic subject/verb/object patterns.
"""

# Concept seed words referenced in some sentences
_concepts = [
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

# Components for generating generic sentences on various topics
_subjects = [
    "The cat",
    "The dog",
    "The teacher",
    "A student",
    "The artist",
    "The engineer",
    "The farmer",
    "The doctor",
    "The parent",
    "The child",
    "The singer",
    "The dancer",
    "The writer",
    "The leader",
    "The traveler",
    "The actor",
    "The scientist",
    "The poet",
    "The builder",
    "The chef",
    "The gamer",
    "The painter",
    "The ranger",
    "The soldier",
    "The pilot",
    "The nurse",
    "The driver",
    "The coder",
    "The blogger",
    "The musician",
]

_verbs = [
    "walks to",
    "runs toward",
    "jumps over",
    "sits beside",
    "looks at",
    "listens to",
    "finds",
    "loses",
    "builds",
    "reads",
    "writes",
    "draws",
    "explores",
    "examines",
    "greets",
    "questions",
    "studies",
    "shares",
    "plays with",
    "ignores",
    "discovers",
    "fixes",
    "throws",
    "carries",
    "enjoys",
    "cooks",
    "drinks",
    "drives to",
    "waits for",
    "dreams about",
]

_objects = [
    "the park",
    "the library",
    "a book",
    "a puzzle",
    "a friend",
    "an idea",
    "a secret",
    "the city",
    "the sky",
    "a game",
    "the ocean",
    "a plan",
    "the mountain",
    "the river",
    "a project",
    "a song",
    "a dance",
    "the computer",
    "the phone",
    "the forest",
    "a cookie",
    "the garden",
    "the movie",
    "a story",
    "the future",
    "the past",
    "the truth",
    "the question",
    "the answer",
    "the road",
]

# Build the dataset
training_texts = []

# Sentences that mention the RFL concept seed words
for i in range(1000):
    concept = _concepts[i % len(_concepts)]
    training_texts.append(f"Statement about {concept} number {i + 1}.")

# Generic short sentences across many topics
for i in range(9000):
    s = _subjects[i % len(_subjects)]
    v = _verbs[i % len(_verbs)]
    o = _objects[i % len(_objects)]
    training_texts.append(f"{s} {v} {o}.")

# Safety check so consumers can rely on the exact size
assert len(training_texts) == 10000

if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(len(training_texts))
    print(training_texts[:5])
    print("...")
    print(training_texts[-5:])
