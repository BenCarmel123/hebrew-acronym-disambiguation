"""Shared pair encoding for cross-encoder training and candidate evaluation."""

import torch

from model.common.pairs import ACR_CLOSE, ACR_OPEN


def encode_pairs(tok, pairs_batch, device, max_len=256,
                 acr_open_id=None, acr_close_id=None):
    """Encode marked context/candidate pairs with the original target-centred crop.

    Empty batches and missing markers retain their original ValueError behavior.
    A target plus candidate longer than the budget is kept, even beyond max_len.
    """
    if acr_open_id is None or acr_close_id is None:
        open_id, close_id = tok.convert_tokens_to_ids([ACR_OPEN, ACR_CLOSE])
        acr_open_id = open_id if acr_open_id is None else acr_open_id
        acr_close_id = close_id if acr_close_id is None else acr_close_id
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    rows = []
    for marked_context, candidate in pairs_batch:
        ctx = tok.encode(marked_context, add_special_tokens=False)
        cand = tok.encode(candidate, add_special_tokens=False)
        o, c = ctx.index(acr_open_id), ctx.index(acr_close_id)
        budget = max_len - 3 - len(cand)
        lo, hi = 0, len(ctx)
        while (hi - lo) > budget:
            left_room, right_room = o - lo, hi - (c + 1)
            if left_room >= right_room and left_room > 0:
                lo += 1
            elif right_room > 0:
                hi -= 1
            else:
                break
        ctx = ctx[lo:hi]
        ids = [cls_id] + ctx + [sep_id] + cand + [sep_id]
        types = [0] * (len(ctx) + 2) + [1] * (len(cand) + 1)
        rows.append((ids, types))
    width = max(len(ids) for ids, _ in rows)
    pad = tok.pad_token_id or 0
    input_ids, attention_mask, token_type_ids = [], [], []
    for ids, types in rows:
        n = width - len(ids)
        input_ids.append(ids + [pad] * n)
        attention_mask.append([1] * len(ids) + [0] * n)
        token_type_ids.append(types + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long, device=device),
    }
