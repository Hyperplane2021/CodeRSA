# CodeRSA: Pragmatic Reranking for LLM Code Generation

This repository hosts the *artifact package* for the paper **"Pragmatic Reasoning improves LLM Code Generation"**.
It is designed for reviewers to verify the **authenticity and integrity of the provided data files** and to inspect a clean, minimal implementation layout of the re-ranking pipeline (CodeRSA).

> **Goal of this repo**: make data verification and inspection easy. Full re-runs are optional and left to interested reviewers to perform with their own compute.

## What you can do here
- ✅ Verify that every data file shipped by us is **complete, unmodified, and valid** (checksum + optional schema checks).
- ✅ Explore a **minimal, readable project layout** for CodeRSA (interfaces & stubs are included; drop-in your code later).
- ✅ (Optional) Reproduce results by plugging in your compute and models; we provide orchestration hooks but do not force full runs.

## Repository Layout
```
.
├── README.md
├── CITATION.cff
├── LICENSE
├── Makefile
├── .gitignore
├── env/
│   ├── requirements.txt
│   └── environment.yml
├── data/
│   ├── MANIFEST.json            # authoritative list of data files + SHA256
│   └── schema/                  # optional JSON/YAML schemas for validation
├── scripts/
│   ├── compute_checksums.py     # produce/refresh SHA256 in MANIFEST.json
│   ├── verify_data.py           # verify data integrity & basic sanity checks
│   └── validate_schema.py       # optional schema-based validation
├── src/
│   └── codersa/
│       ├── __init__.py
│       ├── sampling.py          # candidate sampling interfaces
│       ├── rsa.py               # L0, S1, L1 definitions (interfaces)
│       ├── clustering.py        # paraphrase-equivalence clustering hooks
│       └── reranking.py         # glue: compute scores, rank candidates
└── tests/
    └── test_data_manifest.py    # quick integrity test for the manifest
```
## Quick Start
```bash
# 1) Create and activate env (choose one)
python -m venv .venv && source .venv/bin/activate && pip install -r env/requirements.txt
# or
conda env create -f env/environment.yml && conda activate codersa

# 2) Verify any data you place into ./data
python scripts/compute_checksums.py    # populates/updates data/MANIFEST.json
python scripts/verify_data.py          # verifies checksums & light sanity checks

# 3) (Optional) run tests
pytest -q
```

## Data authenticity & validity
- **Integrity**: `scripts/verify_data.py` re-hashes each file and compares against `data/MANIFEST.json` (SHA256).
- **Completeness**: `MANIFEST.json` is treated as the source of truth. Anything missing or unexpected is reported.
- **Validity (optional)**: if schemas are present in `data/schema/`, `scripts/validate_schema.py` can run structural checks.

## How to add your data (authors)
1. Place your released files under `data/` (or nested subdirs).
2. Run `python scripts/compute_checksums.py` to create/update `MANIFEST.json` with file sizes and SHA256 digests.
3. Commit `MANIFEST.json` alongside your data so reviewers can verify byte-for-byte fidelity.

## Reproduction policy (minimal by default)
We provide clean interfaces under `src/codersa/` for *CodeRSA* components:
- **Literal Listener (L0)**, **Pragmatic Speaker (S1)** with candidate-specific temperatures, and **Pragmatic Listener (L1)**.
- **Paraphrase clustering** with an LLM-based equivalence oracle and cluster-level scoring.
- **Reranking driver** to combine scores and select the best candidate.

The default package ships with **interfaces + reference stubs**. You can drop your own implementation to reproduce results.
We intentionally avoid large heavy checkpoints and lengthy jobs in CI for reviewer convenience.

## Artifact badges (local checks)
- Integrity: `make verify-data` → **PASS/FAIL** summary.
- Manifest consistency: `make checksum` then re-run `make verify-data`.

## Contact
Open an issue or PR on the GitHub repository once you push this package.

---

## Featured dataset: `rsa_with_tokens`

本仓库随附关键验证数据：`data/rsa_with_tokenprior_sum.json`。  
我们提供**可复现的选择逻辑验证器**（与论文实现一致的推选逻辑：以 Topic_RSA 的 softmax[0] 作为分数，并用标准分数化 Prior 设定温度 `τ=exp(-A·z)`，A=1）。

### 运行数据真实性与有效性校验
```bash
# 生成/刷新清单（会计算 SHA256 与大小）
python scripts/compute_checksums.py

# 校验 data/ 与 MANIFEST.json 完整一致
python scripts/verify_data.py

# 运行 rsa-with-tokens 验证逻辑（输出 strict/mean accuracy）
python scripts/verify_rsa_with_tokens.py --file data/rsa_with_tokenprior_sum.json
```
输出形如：
```json
{"strict_accuracy": 0.51, "mean_accuracy": 0.67, "total": 257}
```
> 注：`strict_accuracy` 计算为最佳候选是否满分（accuracy==100.0）的比例；`mean_accuracy` 为最佳候选的平均分（/100）。

---

## Featured dataset: CodeRSA on MBPP (Llama-3 8B Instruct)

**File:** `data/rsa_mbpp_llama3_8b.json`

This dataset contains pragmatic re-ranking outputs computed from the **MBPP** benchmark using **Meta Llama-3 8B Instruct**.
Each instance includes:
- `Prior` — model-estimated prior probabilities for each code candidate  
- `Topic_RSA` — pragmatic topic-level posterior scores  
- `accuracy` — measured functional correctness (0–100)

Use the provided script to verify and recompute summary metrics:

```bash
python scripts/verify_rsa_with_tokens.py --file data/rsa_mbpp_llama3_8b.json
```

Expected output (for reference):
```json
{"strict_accuracy": 0.5953, "mean_accuracy": 0.6462, "total": 257}
```
