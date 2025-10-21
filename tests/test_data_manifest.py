import json, os, hashlib, pathlib

def test_manifest_roundtrip(tmp_path):
    # Ensure manifest structure is acceptable even when empty.
    path = pathlib.Path("data/MANIFEST.json")
    assert path.exists(), "data/MANIFEST.json should exist. Run compute_checksums if missing."
    m = json.loads(path.read_text(encoding="utf-8"))
    assert "spec_version" in m and "files" in m and "base_dir" in m
