# CodeRSA: Pragmatic Re-ranking for LLM Code Generation

This repository provides the **reproducibility and data verification package** for the paper  
**“Pragmatic Reasoning Improves LLM Code Generation.”**

It is designed to allow reviewers and researchers to:
- Verify the authenticity and integrity of released data files.
- Explore a clean, minimal implementation layout of the *CodeRSA* re-ranking pipeline.
- Optionally reproduce the reported results with their own compute and model checkpoints.

---

## 🧩 Repository Overview

```
.
├── README.md
├── CITATION.cff
├── LICENSE
├── Makefile
├── env/
│   ├── requirements.txt
│   └── environment.yml
├── data/
│   ├── rsa_mbpp_llama3_8b.json      # Main dataset: MBPP + Llama-3 8B Instruct
│   ├── MANIFEST.json                # SHA256 checksum manifest
│   └── schema/                      # Optional JSON/YAML schemas for validation
├── scripts/
│   ├── compute_checksums.py         # Generate/update MANIFEST.json
│   ├── verify_data.py               # Verify integrity and completeness
│   ├── verify_rsa_with_tokens.py    # Validate core RSA-with-tokens logic
│   └── validate_schema.py           # Optional schema validation
├── src/
│   └── codersa/
│       ├── __init__.py
│       ├── sampling.py
│       ├── rsa_core.py
│       ├── clustering.py
│       └── reranking.py
└── tests/
    └── test_manifest_integrity.py
```

---

## ⚙️ Environment Setup

You can use either `venv` or `conda`:

```bash
# Option 1: virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt

# Option 2: conda environment
conda env create -f env/environment.yml
conda activate codersa
```

---

## 🧾 Data Authenticity and Validation

All released data files are tracked through `data/MANIFEST.json`, which contains SHA256 checksums and file sizes.

**Integrity check:**
```bash
python scripts/verify_data.py
```

**Manifest regeneration (if you modify data):**
```bash
python scripts/compute_checksums.py
```

A successful validation will output:
```
All files match manifest. ✅ Integrity PASS
```

---

## 📊 Featured Dataset: CodeRSA on MBPP (Llama-3 8B Instruct)

**File:** `data/rsa_mbpp_llama3_8b.json`

This dataset contains pragmatic re-ranking results computed from the **MBPP** benchmark using the **Meta Llama-3 8B Instruct** model.  
Each instance includes:
- `Prior` — model-estimated prior probabilities for each code candidate  
- `Topic_RSA` — pragmatic topic-level posterior scores  
- `accuracy` — functional correctness (0–100)

### Run the RSA-with-Tokens Validation
```bash
python scripts/verify_rsa_with_tokens.py --file data/rsa_mbpp_llama3_8b.json
```

**Expected output:**
```json
{"strict_accuracy": 0.5953, "mean_accuracy": 0.6462, "total": 257}
```

- `strict_accuracy`: proportion of tasks where the top-ranked candidate achieved 100 % accuracy.  
- `mean_accuracy`: average accuracy (in [0, 1]) of the top-ranked candidate.

---

## 🧠 Minimal Reproduction Interface

The folder `src/codersa/` provides modular interfaces for:
- `sampling.py` — candidate generation  
- `rsa_core.py` — L0 / S1 / L1 pragmatic reasoning definitions  
- `clustering.py` — paraphrase-equivalence clustering  
- `reranking.py` — re-ranking pipeline integrating all components  

These files contain interfaces and reference stubs; users can plug in their own implementations for full reproduction.

---

## 🧪 Testing

A minimal integrity test is provided:

```bash
pytest -q
```


---

## 📄 License

This project is released under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.
