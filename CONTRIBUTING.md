# Contributing — dev workflow

## Setup mínimo

```bash
git clone <repo-url> ml-signal-engine
cd ml-signal-engine
python -m venv .venv && source .venv/bin/activate
make install        # pip install -e .[dev]
make precommit-install
```

Si vas a tocar la capa de features que depende de `pandas-ta`, asegurate
de estar en Python **3.12+** y usá:

```bash
make install-py312   # incluye legacy-features
```

## Ciclo de trabajo

```bash
# 1. crear branch
git checkout -b feat/short-descriptive-name

# 2. cambiar código...

# 3. validar localmente
make lint                # ruff
make typecheck           # mypy estricto en src/
make test                # pytest -W error (224 verdes en clean)

# 4. commit (los pre-commit hooks corren ruff + mypy + higiene)
git commit -am "feat(area): qué cambia y por qué"

# 5. push
git push -u origin <branch>
```

CI corre la **suite matrix Python 3.11/3.12** + ruff sobre `src/ tests/ scripts/`
(ver `.github/workflows/test.yml`).

## Convenciones

### Commits

* `feat(area): ...` — nueva funcionalidad.
* `fix(area): ...` — bug fix.
* `refactor(area): ...` — sin cambio funcional.
* `docs(area): ...` — sólo documentación.
* `chore: ...` — tooling, deps.
* `test: ...` — sólo tests.

`area` típicas: `models`, `data`, `training`, `backtest`, `risk`,
`observability`, `tuning`, `connectors`, `ci`.

### Tests

* **Cada fix de bug exige un test que falla sin el fix**.
* **Cada nueva feature exige al menos un test smoke** + uno por edge case.
* Los tests no deben depender de orden ni de side effects globales.
  Usar fixtures `tmp_path`, `monkeypatch`.
* `-W error` está activo en CI: si tu código genera un `DeprecationWarning`
  o `RuntimeWarning`, el test falla. Suprimirlo localmente sólo cuando
  sea inevitable (e.g. dependencias de terceros).

### Estilo

* `ruff format` y `ruff check` (pin en `[dev]`).
* Tipos: `mypy --check-untyped-defs` sobre `src/`. Los tests pueden ser
  no-typed.
* Docstrings: estilo conciso, explicar el **por qué** + invariantes; no
  duplicar la signatura.

### Archivos generados

Todos los siguientes **no se versionan** (ver `.gitignore`):

* `ckpts/`, `checkpoints/` — checkpoints de entrenamiento.
* `optuna_studies/` — SQLite de Optuna.
* `data/*.duckdb` — bases sintéticas / locales.
* `bt_results*.json`, `*_results.json` — outputs de backtest.
* `calibrator_bundle*.json` — curvas isotónicas persistidas.
* `mlruns/` — MLflow.
* `.env` (sin `.example`) — secretos locales.

Para limpiarlos: `make clean-artifacts` (pide confirmación).

## Estructura del repo

Ver [`README.md`](./README.md) sección "Layout del repo". Resumen rápido:

```
src/    ← código de producción (importable)
tests/  ← pytest (espejo de src/)
scripts/← CLIs (no se importan desde src/)
```

**Regla**: nada de `src/` puede importar desde `scripts/` ni `tests/`.

## Roadmap y trazabilidad

* [`AUDIT_TRACKING.md`](./AUDIT_TRACKING.md) — checklist completo (P1-P4 + capas A-G).
  Marcar items cerrados como `[x]` y agregar referencia a tests + commit.
* Nuevos bugs/features grandes: abrir issue, linkear desde el commit.

## Ejecutar tests específicos

```bash
# Todo
make test

# Sólo un archivo
pytest tests/test_backtest.py -W error -v

# Sólo un test
pytest tests/test_models_fixes.py::test_tft_fusion_is_causal_under_future_perturbation -W error -v

# Smoke rápido (sin DDP)
make test-fast

# Con cobertura
make coverage
```

## Benchmarks

```bash
make bench   # target p99 < 5ms CPU
# o con flags custom:
python scripts/benchmark_inference.py --iterations 1000 --mode full \
    --target-p99-ms 3.0 --fail-on-regression --json
```

El target queda documentado en `AUDIT_TRACKING.md::F4`. Cualquier PR que
suba p99 más de 10% debería justificarlo en el mensaje del commit.

## Reportar bugs

Issue template recomendado:

```
**Qué pasó:**
**Qué esperabas:**
**Cómo reproducir:** (comando exacto + commit hash)
**Suite afectada:** (passed / failed / xfailed / errored)
**Logs (si aplica):** ...
```
