# ML-Signal-Engine

Motor híbrido de generación de señales binarias (CALL/PUT, HIGHER/LOWER,
ONETOUCH/NOTOUCH…) para contratos Deriv, con calibración isotónica de
baja latencia, conformal prediction, risk manager y walk-forward
backtesting.

> **Estado**: 35/35 ítems de la auditoría P1-P4 cerrados + capa
> productiva (live inference, backtester, training CLI, Optuna tuner,
> observabilidad, CI). **224 tests** verdes bajo `pytest -W error`.
> Ver [`AUDIT_TRACKING.md`](./AUDIT_TRACKING.md) para el detalle por ítem.

---

## Arquitectura

```
                  ┌────────────────────────────────────────┐
                  │  Deriv WebSocket API v2 (asíncrono)    │
                  └───────────┬────────────────┬───────────┘
                              │ ticks/candles  │ subscribe
                              ▼                ▼
                  ┌──────────────────────┐  ┌──────────────┐
                  │  DuckDBStore         │  │  live loop   │
                  │  (ingest+backfill)   │  │  scripts/    │
                  │  src/connectors/     │  │  infer.py    │
                  └──────────┬───────────┘  └──────┬───────┘
                             │ WindowDataset       │
                             ▼                     │
   ┌─────────────────────────────────────────┐     │
   │  CandleFeatureBuilder / TickFeatureBuilder    │
   │  (causal, switch automático por kind)         │
   └──────────────────────┬──────────────────┘     │
                          ▼                        │
   ┌─────────────────────────────────────────┐     │
   │  Backbone CNN1D → BiLSTM → TFTFusion    │     │
   │  + AssetTimeframeEmbedding              │     │
   │  → MultiContractMultiHorizonHead        │     │
   │    (CALL/PUT × horizons × contracts)    │     │
   └──────────────────────┬──────────────────┘     │
                          ▼                        │
   ┌─────────────────────────────────────────┐     │
   │  PerContractCalibratorBundle (isotónico)│     │
   │  + ConformalBundle (coverage ≥ 1-α)     │     │
   │  + OnlineCalibrationMonitor (drift)     │     │
   └──────────────────────┬──────────────────┘     │
                          ▼                        ▼
                  ┌─────────────────────────────────────┐
                  │  SignalPolicy + RiskManager         │
                  │  → señal JSON (CALL/PUT/NO_TRADE +  │
                  │     sizing, route, regime)          │
                  └──────────────┬──────────────────────┘
                                 ▼
                        backtester / live trading
```

---

## Quickstart (clone → install → run)

### Requisitos

* Python **3.11 o 3.12**. En 3.11 los tests dependientes de `pandas-ta`
  se auto-skipean (no hay wheels publicadas).
* `pip` ≥ 24 / `uv` (opcional).
* ~2 GB de espacio (torch + xgboost + numba).

### Instalación reproducible

```bash
git clone <repo-url> ml-signal-engine
cd ml-signal-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Si querés el stack completo (incluye Optuna, Prometheus, websockets):

```bash
pip install -e ".[full,dev,observability]"
```

Si estás en Python **3.12**, agregá `legacy-features` para correr la
suite que depende de `pandas-ta`:

```bash
pip install -e ".[full,dev,observability,legacy-features]"
```

### Verificación

```bash
make test     # corre la suite completa con -W error
make lint     # ruff
make bench    # benchmark de latencia (target p99 < 5ms CPU)
```

Esperado: **224 passed** en Py 3.11. En 3.12 + `legacy-features`,
**+7 tests** de `feature_generator` e `integrity`.

### Pre-commit hooks (recomendado)

```bash
pre-commit install         # corre ruff + mypy + higiene al commit
pre-commit run --all-files # validación manual
```

---

## Workflows

### 1. Ingestar datos históricos de Deriv

```bash
pip install -e ".[deriv-ingest]"

python -m src.connectors.deriv.ingest \
    --db ./data/market.duckdb \
    --symbol R_100 --symbol R_50 \
    --granularity 60 --candles --ticks \
    --start "2024-01-01" --end "2024-06-30"
```

Crea/actualiza tablas `ticks` y `candles` con upsert + gap detection.

### 2. Entrenar (CPU / single GPU / DDP — auto-detect)

```bash
python scripts/train.py \
    --db ./data/market.duckdb \
    --symbol R_100 --symbol R_50 \
    --kind candles --granularity 60 \
    --window-size 60 --horizons 1 3 5 10 \
    --contracts CALLPUT HIGHERLOWER \
    --epochs 10 --batch-size 128 \
    --device-strategy auto \
    --checkpoint-dir ./ckpts
```

Salidas: `ckpts/best.pt`, `ckpts/last.pt`, `ckpts/calibrator_bundle.json`.

Para DDP multi-GPU:

```bash
python scripts/train.py ... --device-strategy ddp --world-size 4
```

### 3. Hyperparameter tuning (Optuna)

```bash
python scripts/tune.py \
    --target backbone \
    --db ./data/market.duckdb --symbol R_100 \
    --kind candles --granularity 60 \
    --study-name backbone_R_100 --n-trials 30 \
    --max-epochs-per-trial 5 --k-folds 3
```

Storage SQLite reanudable: `optuna_studies/backbone_R_100.db`.

XGBoost meta-learner (sin GPU):

```bash
python scripts/tune.py \
    --target xgboost --xgb-X X.npy --xgb-y y.npy \
    --study-name meta_xgb --n-trials 50
```

### 4. Backtest walk-forward

```bash
python scripts/backtest.py \
    --mode walk-forward \
    --db ./data/market.duckdb --symbol R_100 \
    --kind candles --granularity 60 \
    --window-size 60 --horizons 1 3 --contracts CALLPUT \
    --n-folds 5 --epochs-per-fold 3 \
    --max-drawdown 100 --max-daily-loss 50 \
    --conformal-alpha 0.1 \
    --output ./bt_results.json
```

Reporte JSON con métricas agregadas + per-fold (Sharpe, Sortino, max DD,
win rate, profit factor, breakdown por contrato).

### 5. Inferencia en vivo

```bash
python scripts/infer.py \
    --checkpoint ./ckpts/best.pt \
    --calibrator-bundle ./ckpts/calibrator_bundle.json \
    --app-id $DERIV_APP_ID \
    --symbol R_100 --style candles --granularity 60 \
    --window-size 60 --horizons 1 3 \
    --contracts CALLPUT
```

Stream JSON de señales por stdout. Soporta `--max-iterations N`
para smoke testing / CI.

### 6. Benchmark de latencia

```bash
python scripts/benchmark_inference.py \
    --iterations 1000 --warmup 100 \
    --window-size 60 --num-features 14 \
    --mode full --device cpu --target-p99-ms 5.0 \
    --json
```

Target documentado: **p99 < 5 ms CPU**. `--fail-on-regression` retorna
exit code 1 para gating de CI.

---

## Layout del repo

```
src/
├── connectors/deriv/    # WebSocket client + ingester + DuckDBStore
├── data/                # WindowDataset, FeatureBuilders, labelers, sampler
├── models/              # backbone, heads, calibration, conformal, drift
├── backtest/            # engine, metrics, walk-forward orchestrator
├── risk/                # RiskManager con kill-switches y caps
├── training/            # Trainer auto-detect + tuning con Optuna
├── observability/       # JSON logging + Prometheus metrics
├── features/            # legacy FeatureGenerator (pandas-ta, Py 3.12+)
└── utils/               # integrity + helpers

scripts/
├── train.py             # CLI entrenamiento end-to-end
├── infer.py             # CLI inferencia en vivo
├── backtest.py          # CLI backtest walk-forward / static
├── tune.py              # CLI Optuna (backbone | xgboost)
└── benchmark_inference.py  # CLI benchmark latencia

tests/                   # 224 tests bajo pytest -W error
.github/workflows/test.yml    # CI matrix Py 3.11 / 3.12 + ruff
.pre-commit-config.yaml       # ruff + mypy + higiene
AUDIT_TRACKING.md             # checklist completo P1-P4 + roadmap
```

---

## Variables de entorno

Copiá `.env.example` a `.env` y completá. Las claves opcionales:

| Variable | Para | Default |
|----------|------|---------|
| `DERIV_APP_ID` | Conector WebSocket Deriv | requerido en live/ingest |
| `DERIV_API_TOKEN` | Auth de portfolio/buy/sell | requerido para trading real |
| `DERIV_ENDPOINT` | URL alternativa del WS | `wss://ws.derivws.com/websockets/v3` |
| `LOG_LEVEL` | Nivel global (DEBUG/INFO/WARNING/ERROR) | INFO |
| `PROMETHEUS_PORT` | Puerto para `start_http_server` opcional | 9090 |

---

## Roadmap (lo que falta)

Resumen vivo en [`AUDIT_TRACKING.md`](./AUDIT_TRACKING.md). Items
pendientes de alto leverage:

- **E3** — Trazas OpenTelemetry desde `generate_signal`.
- **C1-C3** — `scripts/train.py --resume <ckpt>` (foundation ya en `Trainer.load_checkpoint`); script DDP-spawn dedicado.
- **D1** — Adaptador con fallback automático entre `FeatureGenerator` (pandas-ta) y `CandleFeatureBuilder`.
- **D2** — Sampler DDP-aware que distribuya también por símbolo.
- **D3** — Caché Parquet de features pre-computadas para escalar a >1M ventanas.

---

## Licencia

Proprietary. Ver `pyproject.toml`.
