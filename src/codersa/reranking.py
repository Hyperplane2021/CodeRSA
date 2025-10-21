"""
Glue code that calls sampling, scoring (L0), clustering, and pragmatic S1/L1 to produce a ranking.
"""
from typing import List, Dict

def rank_candidates(original_instruction: str, candidates: List[str]) -> Dict[str, float]:
    """
    Return a dict candidate -> final score. Highest score is top-1.
    Implement by:
      1) generating/collecting alternative instructions,
      2) clustering equivalent paraphrases,
      3) computing P_L0(c | cluster) and pragmatic speaker over clusters with tau_c,
      4) returning S1(main_cluster | c; tau_c) as the L1 score used for ranking.
    """
    raise NotImplementedError("Provide your ranking pipeline.")
