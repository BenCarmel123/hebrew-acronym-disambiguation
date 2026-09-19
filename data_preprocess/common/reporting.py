"""Write and display the JSON summaries shared by data-processing workflows."""
from __future__ import annotations

import json
import logging
from pathlib import Path

LOG = logging.getLogger("data_preprocess")


def write_summary(out: str, summary: dict, explicit: str | None = None) -> None:
    path = Path(explicit or f"{out}.summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("summary -> %s", path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
