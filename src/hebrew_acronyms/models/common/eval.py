"""Shared open-generation / candidate-select evaluation loop, usable against any LLM
backend. A backend module (src/hebrew_acronyms/models/qwen/eval.py, src/hebrew_acronyms/models/gemini/eval.py, ...) only
needs to supply a `generate_fn(prompt: str) -> str` and call `evaluate()`.
"""
from __future__ import annotations

import random

from tqdm import tqdm

from hebrew_acronyms.models.common.pairs import _normalise

#: Seed for one random generator per evaluation call. Reproducing candidate order
#: requires the same ordered inputs and candidate lists; it is not keyed by item ID.
SHUFFLE_SEED = 42


def candidate_labels(n_candidates: int) -> list[str]:
    """Label every candidate using bijective base 26: A..Z, AA..AZ, BA..."""
    if type(n_candidates) is not int or n_candidates < 0:
        raise ValueError("Candidate count must be a nonnegative integer")
    labels = []
    for number in range(1, n_candidates + 1):
        label = ""
        while number:
            number, remainder = divmod(number - 1, 26)
            label = chr(65 + remainder) + label
        labels.append(label)
    return labels


def build_generate_prompt(acronym: str, sentence: str) -> str:
    return (
        f"בהתחשב במשפט הבא בעברית, מהו הפירוש של ראשי התיבות \"{acronym}\"?\n"
        f"ענה במילים ספורות בלבד, ללא הסבר.\n\n"
        f"משפט: {sentence}\n"
        f"פירוש:"
    )


def build_select_prompt(acronym: str, sentence: str, shuffled_candidates: list[str]) -> str:
    """Request a label using every candidate in the order supplied by the caller."""
    letters = candidate_labels(len(shuffled_candidates))
    options = "\n".join(f"{l}. {c}" for l, c in zip(letters, shuffled_candidates))
    return (
        f"בהתחשב במשפט הבא בעברית, מהו הפירוש של ראשי התיבות \"{acronym}\"?\n"
        f"ענה בסימון של אפשרות אחת בלבד (למשל: A או AA), בלי שום טקסט נוסף.\n\n"
        f"משפט: {sentence}\n\n"
        f"{options}\n\n"
        f"תשובה:"
    )


def parse_letter_choice(response: str, n_candidates: int) -> int | None:
    """Accept exactly one uppercase in-range label after boundary whitespace trim."""
    if not isinstance(response, str) or type(n_candidates) is not int or n_candidates < 1:
        return None
    answer = response.strip()
    if not answer or any(character < "A" or character > "Z" for character in answer):
        return None
    number = 0
    for character in answer:
        number = number * 26 + ord(character) - 64
        if number > n_candidates:
            return None
    return number - 1


def is_correct(response: str, gold: str) -> bool:
    """Legacy substring score, retained for historical callers only; not study scoring."""
    return _normalise(gold) in _normalise(response)


def is_valid(response: str, candidates: list[str]) -> bool:
    """Whether the response matches ANY candidate, not just the gold one."""
    return any(_normalise(c) in _normalise(response) for c in candidates)


def evaluate(rows: list[dict], generate_fn, mode: str = "generate") -> dict:
    """`generate_fn(prompt: str) -> str` — the one thing that differs per LLM backend."""
    n = 0
    correct = 0
    invalid = 0
    # In select mode, each scored row advances this generator. Earlier rows and their
    # candidate counts therefore affect the order shown for later rows.
    rng = random.Random(SHUFFLE_SEED)
    details = []  # per-item record, so errors can be sliced later (by acronym, type, etc.)
    for r in tqdm(rows, desc=f"{mode} eval", unit="item"):
        # This evaluator skips rows with fewer than two candidates or an empty gold.
        # Unlike the encoder evaluators, it does not check the target's text span.
        cands = [c.strip() for c in r["candidates"].split("|") if c.strip()]
        gold = r["gold_expansion"].strip()
        if len(cands) < 2 or not gold:
            continue
        n += 1

        # Generate mode shows no candidates. Select mode shuffles their order and
        # requests a letter; the displayed order is saved with the response.
        shown_order = None  # only meaningful in select mode; logged so the CSV shows
                            # what the model actually saw, not just the row's raw order
        if mode == "generate":
            prompt = build_generate_prompt(r["acronym"], r["sentence"])
            response = generate_fn(prompt)
            correct_item = is_correct(response, gold)
            valid_item = is_valid(response, cands)
        elif mode == "select":
            shuffled = cands[:]
            rng.shuffle(shuffled)
            shown_order = shuffled
            prompt = build_select_prompt(r["acronym"], r["sentence"], shuffled)
            response = generate_fn(prompt)
            choice = parse_letter_choice(response, len(shuffled))
            valid_item = choice is not None
            correct_item = valid_item and shuffled[choice] == gold
        else:
            raise ValueError(f"unknown mode {mode!r}")

        if correct_item:
            correct += 1
        if not valid_item:
            invalid += 1

        details.append({
            "item_id": r.get("item_id", ""),
            "acronym": r["acronym"],
            "gold": gold,
            "candidates": cands,
            "shown_order": shown_order,
            "response": response,
            "correct": correct_item,
            "valid": valid_item,
            "multi_sense_type": r.get("multi_sense_type", ""),
        })

    return {
        "n_items": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "invalid_rate": round(invalid / n, 4) if n else 0.0,
        "details": details,
    }


def write_details_csv(path: str, details: list[dict]) -> None:
    import csv
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "item_id", "acronym", "gold", "candidates", "shown_order", "response",
            "correct", "valid", "multi_sense_type",
        ])
        writer.writeheader()
        for d in details:
            row = dict(d)
            row["candidates"] = " | ".join(row["candidates"])
            row["shown_order"] = " | ".join(row["shown_order"]) if row["shown_order"] else ""
            writer.writerow(row)


def summarize_selection(rows: list[dict], records: list[dict], condition: str) -> dict:
    """Preliminary dev accuracy over every requested ID, including failures/unrun.

    Derive fresh results without rewriting raw artifacts. Missing type_id makes
    macro unavailable; missing gold makes both accuracies unavailable. No alias,
    case or quote folding is applied. Generation is never scored here.
    """
    from collections import Counter, defaultdict
    from hebrew_acronyms.models.common.pairs import validate_ids
    if condition not in {"dictabert", "select", "qwen_select", "gemini_select"}:
        raise ValueError("Automatic dev accuracy is defined only for selection systems")
    validate_ids(rows)
    selected_records = [record for record in records if record.get("condition", condition) == condition]
    validate_ids(selected_records)
    by_id = {record["item_id"]: record for record in selected_records}
    if set(by_id) != {row["item_id"] for row in rows}:
        raise ValueError("Scoring requires exactly one record per requested item; do not drop failures")
    details, types = [], defaultdict(list)
    missing_gold, missing_type = [], []
    for row in rows:
        record = by_id[row["item_id"]]
        selected, index = None, None
        status = record["status"]
        if condition == "dictabert" and status == "ok":
            selected = record.get("selected_candidate")
            index = record.get("selected_index")
        elif condition != "dictabert" and status == "response_received":
            shown = record.get("shown_order")
            index = parse_letter_choice(record.get("raw_response"), len(shown) if isinstance(shown, list) else 0)
            if index is None:
                status = "parse_error"
            else:
                selected = shown[index]
                status = "ok"
        gold = row.get("gold_expansion")
        if not isinstance(gold, str) or not gold.strip():
            missing_gold.append(row["item_id"])
            correct = None
        else:
            correct = status == "ok" and isinstance(selected, str) and selected.strip() == gold.strip()
        type_id = row.get("type_id")
        if not isinstance(type_id, str) or not type_id.strip():
            missing_type.append(row["item_id"])
        else:
            types[type_id].append(correct)
        details.append({"item_id": row["item_id"], "type_id": type_id,
                        "status": status, "raw_status": record["status"],
                        "selected_candidate": selected, "selected_index": index,
                        "correct": correct, "error": record.get("error")})
    n = len(rows)
    per_type = [{"type_id": key, "n_items": len(values),
                 "accuracy": sum(values) / len(values) if None not in values else None}
                for key, values in sorted(types.items())]
    warnings = []
    if missing_gold:
        warnings.append("Accuracy unavailable: missing gold_expansion for " + ", ".join(missing_gold))
    if missing_type:
        warnings.append("Macro unavailable: missing type_id for " + ", ".join(missing_type))
    attempted = sum(record["status"] != "not_run" for record in selected_records)
    partial = any(detail["raw_status"] in {"not_run", "interrupted"} for detail in details)
    if not attempted:
        for group in per_type:
            group["accuracy"] = None
    system = "qwen" if condition == "select" else condition.removesuffix("_select")
    return {"condition": condition, "system": system, "n_items": n,
            "n_attempted": attempted,
            "n_failures": sum(detail["status"] not in {"ok", "not_run"} for detail in details),
            "execution_status": "not_run" if not attempted else "partial" if partial else "completed",
            "n_valid_predictions": sum(detail["status"] == "ok" for detail in details),
            "partial": any(detail["raw_status"] in {"not_run", "interrupted"} for detail in details),
            "micro_accuracy": sum(detail["correct"] for detail in details) / n if n and attempted and not missing_gold else None,
            "macro_accuracy": sum(group["accuracy"] for group in per_type) / len(per_type)
                if per_type and attempted and not missing_gold and not missing_type else None,
            "by_type": per_type, "status_counts": dict(Counter(detail["status"] for detail in details)),
            "warnings": warnings, "details": details}


def inspect_predictions(rows: list[dict], records: list[dict]) -> dict:
    """Inspect only systems present in the artifact, including historical Qwen runs."""
    from hebrew_acronyms.models.common.pairs import explicit_span, mark_span
    conditions = {record["condition"] for record in records}
    selection_conditions = [name for name in ("dictabert", "select", "qwen_select", "gemini_select")
                            if name in conditions]
    metrics = [summarize_selection(rows, records, name) for name in selection_conditions]
    details = {metric["system"]: {detail["item_id"]: detail for detail in metric["details"]}
               for metric in metrics}
    raw = {(record["condition"], record["item_id"]): record for record in records}
    items, disagreements = [], []
    for row in rows:
        item_id = row["item_id"]
        try:
            marked = mark_span(row["sentence"], explicit_span(row))
        except (KeyError, ValueError):
            marked = None
        item = {"item_id": item_id, "type_id": row.get("type_id"), "sentence": row.get("sentence"),
                "marked_sentence": marked, "target_raw": row.get("target_raw"),
                "span": [row.get("span_start"), row.get("span_end")], "gold": row.get("gold_expansion"),
                "candidates": row.get("candidates"), "systems": {}, "failures": {},
                "generation_score_status": "manual_review_unscored"}
        valid = {}
        for metric in metrics:
            system, condition = metric["system"], metric["condition"]
            detail = details[system][item_id]
            record = raw[(condition, item_id)]
            result = {"selection_decoded": detail["selected_candidate"], "selection_status": detail["status"],
                      "selection_correct": detail["correct"], "selection_raw": record.get("raw_response"),
                      "letter_mapping": dict(zip(candidate_labels(len(record.get("shown_order") or [])),
                                                 record.get("shown_order") or [])),
                      "selection_error": detail.get("error")}
            if detail["status"] == "parse_error":
                result["selection_error"] = "Expected one uppercase candidate label in the displayed range"
            if system != "dictabert":
                generation_condition = "generate" if condition == "select" else system + "_generate"
                generation = raw[(generation_condition, item_id)]
                result.update(generation_raw=generation.get("raw_response"), generation_status=generation["status"],
                              generation_error=generation.get("error"))
            if detail["status"] == "ok":
                valid[system] = detail["selected_candidate"]
            item["systems"][system] = result
            for task in ("selection", "generation"):
                if result.get(task + "_error"):
                    item["failures"][system + "_" + task] = result[task + "_error"]
        item["disagreement"] = len({value.strip() for value in valid.values()}) > 1
        item["valid_selections"] = valid
        # Preserve the earlier inspection dictionary keys for Qwen-only consumers.
        enc, qwen = item["systems"].get("dictabert", {}), item["systems"].get("qwen", {})
        item.update(encoder_prediction=enc.get("selection_decoded"), encoder_status=enc.get("selection_status"),
                    encoder_correct=enc.get("selection_correct"))
        item.update(qwen)
        if qwen.get("selection_error"):
            item["failures"]["select"] = qwen["selection_error"]
        items.append(item)
        if item["disagreement"]:
            disagreements.append(item)
    return {"items": items, "metrics": metrics, "disagreements": disagreements,
            "generation_score_status": "manual_review_unscored"}
