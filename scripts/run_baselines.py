
#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codersa.loader import load_dataset, iter_instances
from codersa.rsa_eval import select_prior_argmax
from codersa.metrics import strict_and_mean

def main():
    ap = argparse.ArgumentParser(description="Baseline: Prior argmax selection.")
    ap.add_argument("--file", default=str(Path(__file__).resolve().parents[1] / "data" / "rsa_mbpp_llama3_8b.json"))
    args = ap.parse_args()

    data = load_dataset(args.file)
    selected = [select_prior_argmax(inst) for inst in data["instances"]]
    strict, mean, total = strict_and_mean(selected)
    print(json.dumps({"strict_accuracy": strict, "mean_accuracy": mean, "total": total}))

if __name__ == "__main__":
    main()
