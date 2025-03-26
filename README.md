# Pragmatic Reasoning Improves LLM Code Generation

This repository contains scripts and methodologies supporting the paper **"Pragmatic Reasoning Improves LLM Code Generation"**.

---

## Repository Structure

```
./
├── README.md
├── codersa_calculation.py
├── coder_reviewer_calculation.py
├── result_analysis.py
└── data/
    ├── bootstrap_output.json
    ├── CodeRSA_Result_Example.json
    └── CoderReviewer_Result_Example.json
```

---

## Environment Setup

To set up the necessary Python environment, run:

```bash
pip install vllm numpy tqdm matplotlib pandas
```

Alternatively, using Conda for environment isolation is recommended:

```bash
conda create -n pragmatic_codegen python=3.10
conda activate pragmatic_codegen
pip install vllm numpy tqdm matplotlib pandas
```

---

## Running Experiments

### Full Experimental Workflow

Run the following scripts sequentially to reproduce the entire experimental workflow:

```bash
python codersa_calculation.py
python coder_reviewer_calculation.py
python result_analysis.py
```

### Inspecting Precomputed Example Results

If you prefer directly inspecting precomputed results without running the full workflow:

```bash
python result_analysis.py data/CoderReviewer_Result_Example.json data/CodeRSA_Result_Example.json
```

---

## Analysis and Results

Results and analysis outputs can be found in the `data/` directory. The key outputs include:
- `bootstrap_output.json`: Questions with candidates.
- `CodeRSA_Result_Example.json`: Example results for the CodeRSA method.
- `CoderReviewer_Result_Example.json`: Example results for the Coder-Reviewer&Coder method.

Use `result_analysis.py` to visualize and analyze these outputs.

---

