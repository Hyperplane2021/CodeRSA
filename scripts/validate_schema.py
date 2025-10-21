#!/usr/bin/env python3
import os, json, sys
from pathlib import Path
from jsonschema import validate, ValidationError

BASE = Path(__file__).resolve().parents[1] / "data"
SCHEMA_DIR = BASE / "schema"

def main():
    if not SCHEMA_DIR.exists():
        print("No schema directory present; skipping schema validation.")
        return

    # Example convention: *.schema.json applies to files with same stem in data/
    for schema_path in SCHEMA_DIR.glob("*.schema.json"):
        stem = schema_path.stem.replace(".schema","")
        target = BASE / f"{stem}.json"
        if not target.exists():
            print(f"[SKIP] No target found for schema {schema_path.name}")
            continue
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        data = json.loads(target.read_text(encoding="utf-8"))
        try:
            validate(instance=data, schema=schema)
            print(f"[OK] {target.name} validated against {schema_path.name}")
        except ValidationError as e:
            print(f"[INVALID] {target.name} -> {e.message}")
            sys.exit(1)
    print("Schema validation complete.")

if __name__ == "__main__":
    main()
