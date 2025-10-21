
#!/usr/bin/env python3
"""Compute Coder–Reviewer prompt logprob scores for each candidate using vLLM."""
from __future__ import annotations
import argparse, json, re, textwrap
from pathlib import Path
from vllm import LLM, SamplingParams

def clean_code(block: str) -> str:
    import re
    block = re.sub(r'#.*$', '', block, flags=re.MULTILINE)
    block = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '', block, flags=re.MULTILINE)
    block = block.strip()
    block = re.sub(r'\n\s*\n', '\n', block)
    return block

def rename_function(text: str) -> str:
    return re.sub(r'def\s+([^\(]+)\(', 'def f(', text)

def extract_docstring(code: str) -> str | None:
    m = re.search(r'(""".*?"""|\'\'\'.*?\'\'\')', code, re.DOTALL)
    if not m: return None
    doc = m.group(1)
    if doc.startswith('"""') and doc.endswith('"""'): doc = doc[3:-3]
    if doc.startswith("'''") and doc.endswith("'''"): doc = doc[3:-3]
    import textwrap as _tw
    return _tw.dedent(doc)

def get_start_index(outputs):
    start = 0
    for n, tok in enumerate(outputs[0].prompt_logprobs):
        if tok is not None:
            vals = list(tok.values())
            if len(vals)>=1 and vals[0].decoded_token == '""' and list(outputs[0].prompt_logprobs[n+1].values())[0].decoded_token == '"':
                start = n+2
                break
    return start

def format_for_reviewer(f: str, input_prompt: str) -> str:
    clean_f = rename_function(clean_code(f))
    comment = extract_docstring(input_prompt) or ""
    extra = "\n# write a docstring for the above function\n"
    return clean_f + extra + '""" ' + comment + '"""'

def format_for_coder(f: str, input_prompt: str) -> str:
    clean_f = rename_function(clean_code(f))
    comment = extract_docstring(input_prompt) or ""
    return '""" ' + comment + '"""\n' + clean_f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args()

    data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    llm = LLM(model=args.model)
    params = SamplingParams(n=1, temperature=1.0, max_tokens=1, prompt_logprobs=1)

    for q in data:
        prompt = q.get("input_prompt","")
        for cand in q.get("output", []):
            f = cand.get("Function","")
            try:
                r_in = format_for_reviewer(f, prompt)
                out = llm.generate(r_in, params)
                start = get_start_index(out)
                reviewer_prob = [list(tok.values())[0].logprob for tok in out[0].prompt_logprobs[start:-2]]
                cand["Reviewer_prob"] = float(sum(reviewer_prob))
            except Exception:
                cand["Reviewer_prob"] = 0.0
            try:
                c_in = format_for_coder(f, prompt)
                out = llm.generate(c_in, params)
                coder_prob = [list(tok.values())[0].logprob for tok in out[0].prompt_logprobs[1:]]
                cand["Coder_prob"] = float(sum(coder_prob))
            except Exception:
                cand["Coder_prob"] = 0.0
            cand["Coder_Reviewer_prob"] = cand.get("Coder_prob",0.0) + cand.get("Reviewer_prob",0.0)

    Path(args.outfile).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote coder-reviewer scored JSON to {args.outfile}")

if __name__ == "__main__":
    main()
