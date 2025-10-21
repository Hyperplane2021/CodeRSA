
from __future__ import annotations
from typing import Dict, Any
import numpy as np

def _softmax(x):
    x = np.array(x, dtype=float)
    e = np.exp(x - np.max(x))
    return e / e.sum()

def select_codersa_with_tokens(inst: Dict[str, Any], A: float = 1.0) -> Dict[str, Any]:
    outputs = inst.get("output", [])
    if not outputs:
        return {}
    priors = np.array([float(o.get("Prior", 0.0)) for o in outputs], dtype=float)
    sd = priors.std()
    if sd <= 1e-12:
        z_all = (priors - priors.mean()) / 1.0
    else:
        z_all = (priors - priors.mean()) / sd

    best = None
    best_score = -1e300
    for idx, cand in enumerate(outputs):
        z = z_all[idx]
        tau = float(np.exp(-A * z))
        topic = np.array(cand.get("Topic_RSA", []), dtype=float)
        if topic.size == 0:
            continue
        p = _softmax(topic / tau)
        score = float(p[0])
        if score > best_score:
            best_score = score
            best = cand
    return best or {}

def select_prior_argmax(inst: Dict[str, Any]) -> Dict[str, Any]:
    outputs = inst.get("output", [])
    if not outputs:
        return {}
    best = max(outputs, key=lambda o: float(o.get("Prior", 0.0)))
    return best
