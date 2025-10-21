
#!/usr/bin/env python3
"""Execute unit tests for each candidate and record pass ratio."""
from __future__ import annotations
import argparse, json, re, typing, math
from pathlib import Path

def extract_function_name(prompt: str) -> str | None:
    m = re.search(r'def\s+(\w+)\(', prompt)
    return m.group(1) if m else "candidate"

def format_test(test_code, func_name: str):
    if isinstance(test_code, str):
        lines = test_code.splitlines()
    else:
        lines = []
        for item in (test_code or []):
            if isinstance(item, str):
                lines.extend(item.splitlines())
    cases = []
    in_check = False
    for line in lines:
        if re.match(r'\s*def\s+check\s*\(\s*candidate\s*\)\s*:', line):
            in_check = True
            continue
        if in_check and re.match(r'\s*def\s+\w+\s*\(', line):
            break
        m = re.match(r'\s*assert\s+(.+)', line)
        if m:
            expr = re.sub(r'\bcandidate\b', func_name, m.group(1))
            cases.append('assert ' + expr)
    return cases

def run_single(func_str: str, test_stmt: str) -> bool:
    local_env = {"re": re, "math": math, "typing": typing, "List": typing.List, "Tuple": typing.Tuple}
    exec(func_str, local_env)
    code_obj = compile(test_stmt, "<test>", "exec")
    exec(code_obj, local_env)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    for q in data:
        func_name = extract_function_name(q.get("input_prompt",""))
        tests = q.get("test_code", "")
        cases = format_test(tests, func_name)
        for cand in q.get("output", []):
            passed = 0
            errors = []
            for stmt in cases:
                try:
                    run_single(cand["Function"], stmt)
                    passed += 1
                except AssertionError:
                    errors.append("assert-failed")
                except Exception as e:
                    errors.append(f"error: {type(e).__name__}")
            total = max(1, len(cases))
            cand["pass"] = round(100.0 * passed / total, 2)
            cand["error"] = errors if errors else ["pass"]
    Path(args.outfile).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote tested JSON to {args.outfile}")

if __name__ == "__main__":
    main()
