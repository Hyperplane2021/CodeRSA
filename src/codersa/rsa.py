"""
RSA components (interfaces) for CodeRSA:
- L0: literal listener P_L0(c | i)
- S1: pragmatic speaker over alternatives, using candidate-specific temperatures
- L1: pragmatic listener that selects among candidates w.r.t. original instruction
"""
from typing import Dict, List
import math

def literal_listener_scores(candidates: List[str], instructions: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Return nested dict mapping candidate -> instruction -> score (e.g., token-prob or pseudo-likelihood).
    Implement with your scoring backend.
    """
    raise NotImplementedError("Provide your scoring implementation.")

def speaker_distribution_over_clusters(pl0_cluster: Dict[str, float], tau: float) -> Dict[str, float]:
    """Softmax over clusters with temperature tau (1/tau applied to logits)."""
    # pl0_cluster: cluster_id -> P_L0(c | Ck)
    # Convert to log domain, apply 1/tau, then softmax
    keys = list(pl0_cluster.keys())
    if not keys:
        return {}
    logits = [math.log(max(pl0_cluster[k], 1e-40)) / max(tau, 1e-8) for k in keys]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    Z = sum(exps)
    return {k: exps[i] / Z for i, k in enumerate(keys)}
