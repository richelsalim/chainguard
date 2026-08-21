.DEFAULT_GOAL := help
PY ?= python
DATA ?= data/synthetic.xlsx
SCENARIO ?= baseline

.PHONY: help install synth profile optimize simulate network benchmark test lint fmt cover app clean all

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with all extras
	$(PY) -m pip install -e ".[all]"

synth: ## Generate a synthetic workbook (no real data is used anywhere)
	$(PY) -m chainguard.cli synth --out $(DATA)

profile: ## Validate a workbook against the schema contract
	$(PY) -m chainguard.cli profile --data $(DATA)

optimize: ## Solve one scenario and compare greedy / repair / MILP
	$(PY) -m chainguard.cli optimize --data $(DATA) --scenario $(SCENARIO)

simulate: ## Monte Carlo service levels for the optimal plan
	$(PY) -m chainguard.cli simulate --data $(DATA) --scenario $(SCENARIO)

network: ## Graph statistics, chokepoints and end-to-end paths
	$(PY) -m chainguard.cli network --data $(DATA) --scenario $(SCENARIO)

benchmark: ## Full head-to-head across all scenarios -> artifacts/
	$(PY) -m chainguard.cli benchmark --data $(DATA)

app: ## Launch the Dash control tower on http://127.0.0.1:8050
	$(PY) app/dashboard.py --data $(DATA)

test: ## Run the test suite
	$(PY) -m pytest

cover: ## Run tests with a coverage report
	$(PY) -m pytest --cov=chainguard --cov-report=term-missing --cov-report=xml

lint: ## Lint with ruff
	$(PY) -m ruff check src tests app

fmt: ## Auto-fix lint issues
	$(PY) -m ruff check --fix src tests app

all: synth test lint benchmark ## Generate data, test, lint, benchmark

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -f artifacts/benchmark.* $(DATA)
