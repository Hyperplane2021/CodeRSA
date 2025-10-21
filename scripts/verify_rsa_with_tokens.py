#!/usr/bin/env python3
import json, argparse
import numpy as np
from pathlib import Path

def softmax(x):
    x = np.array(x, dtype=float)
    e = np.exp(x - np.max(x))
    return e / e.sum()

def main():
    ap = argparse.ArgumentParser(description="Validate rsa_with_tokens data integrity & compute strict/mean accuracy.")
    ap.add_argument("--file", default=str((Path(__file__).resolve().parents[1] / "data" / "rsa_mbpp_llama3_8b.json")), help="Path to JSON data file")
    args = ap.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = data["instances"]
    A = 1.0  # 温度强度，推荐固定为 1
    correct, total, acc_sum = 0, 0, 0.0

    for task in instances:
        priors = np.array([f["Prior"] for f in task["output"]], dtype=float)
        mu, sd = priors.mean(), priors.std() if priors.std()>1e-12 else 1.0
        z_all = (priors - mu)/sd

        best, best_score = None, -1e300
        for idx, func in enumerate(task["output"]):
            z = z_all[idx]
            tau = np.exp(-A*z)
            topic = np.array(func["Topic_RSA"], dtype=float) / tau
            p_first = softmax(topic)[0]
            if p_first > best_score:
                best_score, best = p_first, func

        if best and best.get("accuracy", 0.0) == 100.0:
            correct += 1
        total += 1
        acc_sum += float(best.get("accuracy", 0.0))/100.0

    strict_acc = correct/total if total else 0.0
    mean_acc = acc_sum/total if total else 0.0
    print(json.dumps({"strict_accuracy": strict_acc, "mean_accuracy": mean_acc, "total": total}, ensure_ascii=False))

if __name__ == "__main__":
    main()
