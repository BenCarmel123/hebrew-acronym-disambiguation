"""Load the base DictaBERT tokenizer and encoder without a task-specific head."""
#: Hugging Face id for DictaBERT (base, ~184M params, Hebrew).
MODEL_ID = "dicta-il/dictabert"

#: Pin a commit sha here for reproducible runs. None takes whatever the hub serves today.
REVISION = None


def build_model(model_id=MODEL_ID, revision=REVISION):
    """-> (tokenizer, encoder); import the loading stack only when called."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    enc = AutoModel.from_pretrained(model_id, revision=revision)
    return tok, enc
