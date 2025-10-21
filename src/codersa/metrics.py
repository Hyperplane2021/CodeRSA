
from __future__ import annotations
from typing import Iterable, Dict, Any, Tuple

def strict_and_mean(selected_items: Iterable[Dict[str, Any]]) -> Tuple[float, float, int]:
    total = 0
    strict = 0
    acc_sum = 0.0
    for item in selected_items:
        total += 1
        acc = float(item.get("accuracy", 0.0))
        if acc == 100.0:
            strict += 1
        acc_sum += acc / 100.0
    if total == 0:
        return 0.0, 0.0, 0
    return strict / total, acc_sum / total, total
