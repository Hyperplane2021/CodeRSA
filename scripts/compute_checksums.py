
#!/usr/bin/env python3
import os, json, hashlib, time
from pathlib import Path

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
    files = []
    for p in BASE.rglob("*"):
        if p.is_file() and p.name != "MANIFEST.json":
            rel = p.relative_to(BASE).as_posix()
            files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256sum(p)})
    manifest = {"spec_version": "1.0", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "base_dir": "data", "files": sorted(files, key=lambda x: x["path"])}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest with {len(files)} file(s): {MANIFEST}")

if __name__ == "__main__":
    main()
