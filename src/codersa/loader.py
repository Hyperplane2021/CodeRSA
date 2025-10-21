
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Iterable

def load_dataset(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "instances" in data and isinstance(data["instances"], list), "Malformed dataset JSON: missing 'instances' list"
    return data

def iter_instances(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for inst in data["instances"]:
        yield inst
