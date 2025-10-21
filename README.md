
# CodeRSA: Reproducible Implementation and Verification Package

This repository provides a self-contained reproducibility package for **CodeRSA**, including end-to-end scripts to generate candidates, run unit tests, compute Coder–Reviewer scores, and evaluate RSA-with-tokens. All instructions are in English and suitable for double-blind review.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
python scripts/compute_checksums.py
python scripts/verify_data.py
python scripts/verify_rsa_with_tokens.py --file data/rsa_mbpp_llama3_8b.json --A 1.0
```

## End-to-End Pipeline (Generation → Testing → Coder–Reviewer → RSA)
See `scripts/pipeline/*.py`:
- `generate_llama3.py` (GPU + vLLM)
- `test_candidates.py`
- `coder_reviewer.py`

## Visualization
```bash
python scripts/plot_results.py
```
Generates `results/accuracy_comparison.png` using verified CodeRSA accuracy and reference baselines from `results/summary_baselines.json`.


## Implementation Details (Author-Provided)

For transparency, we include the original author-provided reference scripts used during development.
These are not required to run the minimal verification pipeline, but they document implementation choices.

Located at `scripts/implementation/`:
- `rsa_mbpp_impl.py`
- `llm_cluster_prior.py`
- `testing_harness.py`
- `generation_llama3_raw.py`
- `coder_reviewer_calc_raw.py`
