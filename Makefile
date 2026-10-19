.PHONY: status verify train-plan prepare-data

PY := .venv/bin/python

# Default action is CPU-only and never submits a job.
status:
	$(PY) scripts/audit.py

verify:
	$(PY) scripts/audit.py --require-ready --verify-downloads

train-plan:
	$(PY) scripts/train.py

prepare-data:
	$(PY) scripts/download_assets.py --group ulvr --train-shards 12
	$(PY) scripts/download_assets.py --group benchmarks
	$(PY) scripts/download_assets.py --group model
	$(PY) scripts/prepare_ulvr.py
	$(PY) scripts/prepare_benchmarks.py
