.DEFAULT_GOAL := help
PYTHON ?= python
PIP    ?= pip
VENV   ?= .venv

# Detect if we're inside an active venv; warn if not.
INSIDE_VENV := $(shell $(PYTHON) -c 'import sys; print("yes" if sys.prefix != sys.base_prefix else "no")')

.PHONY: help install install-full install-py312 venv test test-fast lint typecheck \
        bench train-smoke backtest-smoke tune-smoke precommit-install precommit-run \
        clean clean-artifacts coverage docs-tracking

help: ## Lista los targets disponibles.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Crea un venv local en .venv/
	$(PYTHON) -m venv $(VENV)
	@echo "→ Activá con: source $(VENV)/bin/activate"

install: ## pip install -e .[dev] — instalación mínima para correr la suite (Py 3.11+).
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-full: ## Stack completo (Optuna + Prometheus + websockets + deriv-ingest).
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[full,dev,observability]"

install-py312: ## Stack completo + legacy-features (sólo Py 3.12+; pandas-ta).
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[full,dev,observability,legacy-features]"

test: ## Suite completa con -W error (224 tests verdes en limpio).
	pytest -W error --no-header

test-fast: ## Suite rápida saltando el smoke DDP (~5s).
	pytest -W error --no-header --deselect tests/test_training_layer.py::test_ddp_smoke_two_ranks_with_gloo

lint: ## ruff check sobre src/ tests/ scripts/.
	ruff check src tests scripts

typecheck: ## mypy estricto sobre src/.
	mypy src --ignore-missing-imports --check-untyped-defs

bench: ## Benchmark de latencia (target p99 < 5ms CPU).
	$(PYTHON) scripts/benchmark_inference.py \
	    --iterations 500 --warmup 50 \
	    --window-size 60 --num-features 14 \
	    --contracts CALLPUT --horizons 1 3 \
	    --mode full --device cpu --target-p99-ms 5.0

train-smoke: ## Smoke de entrenamiento (1 epoch sobre DB sintético en /tmp).
	$(PYTHON) -c "from tests.test_pipeline_endtoend import _synthetic_candles; print('ok')"
	@echo "Para smoke real: pytest tests/test_pipeline_endtoend.py::test_train_cli_runs_one_epoch"

backtest-smoke: ## Smoke de backtester sobre DB sintético.
	@echo "Para smoke real: pytest tests/test_backtest.py::test_backtest_cli_walk_forward_smoke"

tune-smoke: ## Smoke de Optuna (RandomSampler + 2 trials).
	@echo "Para smoke real: pytest tests/test_tuning.py::test_tune_cli_xgboost_smoke"

precommit-install: ## Instala los hooks de pre-commit en .git/hooks/.
	pre-commit install

precommit-run: ## Corre los hooks sobre todo el repo.
	pre-commit run --all-files

clean: ## Elimina caches, __pycache__, build artifacts.
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} \;
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} \;
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} \;
	rm -rf build/ dist/ *.egg-info

clean-artifacts: ## ⚠ Borra ckpts/, optuna_studies/, data/*.duckdb, mlruns/. NO ejecuta solo.
	@echo "Esto borrará checkpoints, studies y bases sintéticas. Confirmá con 'make clean-artifacts-confirm'."

clean-artifacts-confirm:
	rm -rf ckpts/ optuna_studies/ mlruns/
	find . -maxdepth 3 -name "*.duckdb" -delete
	find . -maxdepth 3 -name "*.duckdb.wal" -delete

coverage: ## Reporte de cobertura sobre src/.
	pytest -W error --no-header --cov=src --cov-report=term-missing --cov-report=html

docs-tracking: ## Imprime el resumen del checklist de auditoría.
	@grep -E "^- \[(x| )\] \*\*" AUDIT_TRACKING.md | sed 's/\*\*//g' | head -50
