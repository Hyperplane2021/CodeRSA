# Convenience targets

.PHONY: setup checksum verify-data test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r env/requirements.txt

checksum:
	python scripts/compute_checksums.py

verify-data:
	python scripts/verify_data.py

test:
	pytest -q
