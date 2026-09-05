"""Load DictaBERT: the pinned Hebrew encoder and its tokenizer.

    from model import build_model
    tok, encoder = build_model()

Nothing else. No head, no loss, no training.
"""
from transformers import AutoModel, AutoTokenizer

#: Hugging Face id for DictaBERT (base, ~184M params, Hebrew).
MODEL_ID = "dicta-il/dictabert"

#: Pin a commit sha here for reproducible runs. None takes whatever the hub serves today.
REVISION = None


def build_model(model_id=MODEL_ID, revision=REVISION):
    """-> (tokenizer, encoder)."""
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    enc = AutoModel.from_pretrained(model_id, revision=revision)
    return tok, enc


if __name__ == "__main__":
    tok, enc = build_model()
    print(f"loaded {MODEL_ID}")
    print(f"  vocab size:  {len(tok)}")
    print(f"  hidden size: {enc.config.hidden_size}")
    print(f"  parameters:  {sum(p.numel() for p in enc.parameters()):,}")
