# Reflexive Fractal Logic (RFL)

This folder contains experiments exploring **Reflexive Fractal Logic**—a notion that proposes
meaning emerges from repeated patterns and self-reference rather than fixed rules.
RFL networks echo internal representations between layers so that concepts converge
through resonance and compression.

## Concept Overview
- RFL treats truth as the recurrence of approximations across layers.
- Compression of representations is valued as insight.
- Layers feed their outputs back, creating resonance that guides learning.

For a deeper explanation see `reflexive_fractal_logic.md` which introduces the idea,
listing core tenets such as "Approximation Is Structure" and "Resonant Convergence."【F:RFL/reflexive_fractal_logic.md†L1-L35】

## Documentation (`.md` files)
- **`reflexive_fractal_logic.md`** – philosophical overview describing how
  rules arise from nested reflections rather than axioms.
- **`rfl_custom_loss.md`** – outlines a loss function rewarding resonance and
  compression between layers; includes the `rfl_loss` definition.【F:RFL/rfl_custom_loss.md†L1-L35】
- **`rfl_model_display.md`** – shows the full PyTorch model code for an RFL network.【F:RFL/rfl_model_display.md†L1-L40】
- **`rfl_neural_architectures.md`** – discusses how RFL ideas inspire new
  neural architectures and lists improvements beyond stacked transformers.【F:RFL/rfl_neural_architectures.md†L1-L21】
- **`rfl_training_concepts.md`** – suggests training data designs such as
  multi-modal analogy pairs and echo reinforcement datasets.【F:RFL/rfl_training_concepts.md†L1-L40】
- **`rfl_training_guide.md`** – a short guide summarizing the network and
  giving tips for multi-context training.【F:RFL/rfl_training_guide.md†L1-L40】

## Python Scripts (`.py` files)
- **`model.py`** – minimal RFL network with stacked `RFLBlock`s that blend
  current input and past resonance. The `RFLNet` class returns a sigmoid score.【F:RFL/model.py†L20-L49】
- **`rfl_full_project.py`** – self-contained file combining the model, custom
  loss, and a `main()` routine demonstrating a training loop.【F:RFL/rfl_full_project.py†L1-L40】【F:RFL/rfl_full_project.py†L50-L108】
- **`forward.py`** – extracted `forward` routine returning optional resonance
  stack during inference.【F:RFL/forward.py†L1-L12】
- **`rfl_loss.py`** – reusable function computing task loss plus resonance
  penalty across layers.【F:RFL/rfl_loss.py†L1-L24】
- **`rfl_main_with_training_and_save.py`** – trains on example text and saves a
  checkpoint `rfl_trained.pt`.【F:RFL/rfl_main_with_training_and_save.py†L50-L108】
- **`rfl_main_with_training_data.py`** – similar to the above but with a
  simpler training loop for three data categories. It prints predictions after
  training.
- **`rfl_model.py`** – lightweight version of the network used by tests and
  examples.【F:RFL/model.py†L1-L49】
- **`rfl_test_echo_examples.py`** – runs inference on short phrases to see the
  model’s raw predictions.【F:RFL/rfl_test_echo_examples.py†L46-L69】
- **`rfl_test_with_pretrained.py`** – loads `rfl_trained.pt` if available and
  evaluates the same phrases.【F:RFL/rfl_test_with_pretrained.py†L44-L75】
- **`test_rfl_model.py`** – simple `unittest` ensuring the model’s forward pass
  outputs a 0‑1 score.【F:RFL/test_rfl_model.py†L1-L22】
- **`usage_example.py`** – short snippet showing how to instantiate
  `RFLNet` and run a dummy input.【F:RFL/usage_example.py†L1-L13】
- **`usage_1_example.py`** – illustrates a training loop using the custom loss
  with resonance tracking.【F:RFL/usage_1_example.py†L1-L17】

The file `rfl_trained.pt` stores weights saved by the training script.

