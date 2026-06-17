.PHONY: install data backtest demo test lint type check fmt clean

install:
	python -m pip install -e ".[dev]"

# Full empirical run (needs network to the data hosts).
backtest:
	python -m whale_clone

# Offline pipeline smoke test (synthetic data, no network). NOT a market claim.
demo:
	python -m whale_clone --demo

# Fetch + cache data without running the verdict (warms the Parquet cache).
data:
	python -c "from whale_clone.config import load_settings; from whale_clone.pipeline import load_data; load_data(load_settings(), refresh=True); print('cached')"

test:
	python -m pytest

lint:
	ruff check .
	ruff format --check .

type:
	mypy

fmt:
	ruff format .
	ruff check --fix .

# What CI runs.
check: lint type test

clean:
	rm -rf .cache .pytest_cache .mypy_cache **/__pycache__ build dist *.egg-info
