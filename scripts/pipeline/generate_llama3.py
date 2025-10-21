
#!/usr/bin/env python3
"""Generate code candidates with vLLM (Llama-3 8B) for MBPP/HumanEval."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from datasets import load_dataset
from vllm import LLM, SamplingParams

def extract_function_name(prompt: str) -> str | None:
    m = re.search(r'def\s+(\w+)\(', prompt)
    return m.group(1) if m else None

def extract_between_markers(content: str) -> str:
    start_marker = "###Code Start###\n"
    end_markers = ["###Code End###", "### Code End ###"]
    if not content.startswith(start_marker):
        return ""
    end_index = -1
    for marker in end_markers:
        end_index = content.find(marker)
        if end_index != -1:
            break
    if end_index == -1:
        return ""
    lines = content[len(start_marker):end_index].strip().split("\n")
    import re as _re
    pat = _re.compile(r'^[ ]{2,}return')
    last_return = -1
    for i, line in enumerate(lines):
        if pat.search(line):
            last_return = i
    if last_return == -1:
        return ""
    return "\n".join(lines[: last_return + 1])

def load_benchmark(name: str):
    name = name.lower()
    if name == "humaneval":
        ds = load_dataset("openai/openai_humaneval")
        records = []
        for i, row in enumerate(ds["test"]):
            records.append({
                "task_id": f"HumanEval/{i}",
                "input_prompt": row["prompt"],
                "test_code": row["test"],
                "output": []
            })
        return records
    elif name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized")
        records = []
        for i, row in enumerate(ds["test"]):
            prompt = row["text"]
            records.append({
                "task_id": f"MBPP/{row.get('task_id', i)}",
                "input_prompt": prompt,
                "test_code": row.get("test_list", row.get("test", "")),
                "output": []
            })
        return records
    else:
        raise ValueError("Unsupported dataset: " + name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mbpp","humaneval"], required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    tasks = load_benchmark(args.dataset)
    llm = LLM(model=args.model)
    params = SamplingParams(n=20, temperature=args.temperature, max_tokens=args.max_tokens)

    for q in tasks:
        fname = extract_function_name(q["input_prompt"]) or "f"
        for _ in range(args.n):
            response = llm.generate("###Code Start###\n" + q["input_prompt"], params)
            for cand in response[0].outputs:
                func = extract_between_markers(response[0].prompt + cand.text)
                if not func:
                    continue
                idx = len(q["output"])
                q["output"].append({
                    "Function_id": f"{fname}_{idx:02d}",
                    "Function": func,
                    "tokens_logprob": 0.0
                })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"Wrote generations: {args.out}")

if __name__ == "__main__":
    main()
