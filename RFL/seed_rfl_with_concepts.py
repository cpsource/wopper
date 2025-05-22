import torch


def _call_encode(encode_fn, concept, input_dim):
    """Call ``encode_fn`` with a flexible signature.

    Some encoding utilities expect both the concept and a target dimension,
    while others (e.g. ``BERTFrontend.encode``) only need the concept text.
    This helper tries the two-argument call first and falls back to a
    single-argument call if necessary.
    """
    try:
        return encode_fn(concept, input_dim)
    except TypeError:
        return encode_fn(concept)

def seed_rfl_with_concepts(model, concept_list, encode_fn):
    """
    Seeds the first RFLBlock in the model with concept vectors.

    Args:
        model: An instance of RFLNet
        concept_list: List of strings representing high-level concepts
        encode_fn: A function that converts a concept string to a vector (e.g., encode_text_to_vector)
    """
    first_block = model.blocks[0]
    num_neurons = first_block.proj_input.out_features
    input_dim = first_block.proj_input.in_features

    with torch.no_grad():
        for i, concept in enumerate(concept_list):
            if i >= num_neurons:
                print(f"Warning: More concepts ({len(concept_list)}) than neurons ({num_neurons}). Truncating.")
                break
            vec = _call_encode(encode_fn, concept, input_dim)
            first_block.proj_input.weight[i].copy_(vec)
            # Optional: encourage early activation
            first_block.proj_input.bias[i] = 1.0
