#!/usr/bin/env python3
import os, json, hashlib, sys
from pathlib import Path
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1] / "data"
MANIFEST = BASE / "MANIFEST.json"

def sha256sum(path, buf_size=1<<20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    if not MANIFEST.exists():
        print("ERROR: data/MANIFEST.json not found. Run scripts/compute_checksums.py first.", file=sys.stderr)
        sys.exit(2)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    listed = {entry["path"]: entry for entry in manifest.get("files", [])}

    ok = True

    # Check for missing files listed in the manifest
    for rel, entry in listed.items():
        path = BASE / rel
        if not path.exists():
            ok = False
            print(f"[MISSING] {rel}")
            continue
        actual_size = path.stat().st_size
        if actual_size != entry["size"]:
            ok = False
            print(f"[SIZE MISMATCH] {rel}: manifest={entry['size']} actual={actual_size}")
        actual_hash = sha256sum(path)
        if actual_hash != entry["sha256"]:
            ok = False
            print(f"[HASH MISMATCH] {rel}: manifest={entry['sha256']} actual={actual_hash}")

    # Check for unexpected files not declared in the manifest
    found = set()
    for p in BASE.rglob("*"):
        if p.is_file() and p.name != "MANIFEST.json":
            rel = p.relative_to(BASE).as_posix()
            found.add(rel)
            if rel not in listed:
                ok = False
                print(f"[UNDECLARED] {rel} not present in MANIFEST.json")

    if ok:
        print("All files match manifest. ✅ Integrity PASS")
        sys.exit(0)
    else:
        print("Integrity checks found issues. ❌", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
