"""Platform-neutral parsing for structured training log records."""

from __future__ import annotations

import re
from typing import Any


def parse_training_metric_line(line: str) -> dict[str, Any] | None:
    match = re.search(r"Step\s+(\d+):\s+(.*)$", line)
    if not match:
        return None
    record: dict[str, Any] = {"step": int(match.group(1))}
    key_map = {
        "loss": "train_loss", "l2": "train_l2_loss", "ce": "train_ce_loss",
        "plan": "train_plan_loss", "plan_aux": "train_plan_aux_loss",
        "emb_var": "train_plan_emb_batch_var", "pred_var": "train_plan_pred_batch_var",
        "emb_norm": "train_plan_emb_norm", "pred_norm": "train_plan_pred_norm",
        "lr": "lr", "steps/sec": "steps_per_sec",
    }
    for key, value in re.findall(r"([A-Za-z0-9_/]+)=([-+0-9.eE]+)", match.group(2)):
        if key in key_map:
            record[key_map[key]] = float(value)
    return record
