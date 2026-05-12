# Auditoría `src/models/` — Checklist de seguimiento

Auditoría priorizada por **impacto en correctitud → robustez → mantenibilidad**.

Leyenda:
- [x] hecho y verificado por test
- [~] hecho, sin test específico todavía
- [ ] pendiente

> Última actualización: commit `a270600` en rama `claude/deriv-websocket-connector-22wxG`.
> Tests verdes en este snapshot: **125/125** (sin warnings bajo `-W error`).

---

## P1 — Críticos (bugs, fugas o riesgos de runtime)

- [x] **1. `tft_attention.py:130-134` — Doble especificación de causalidad mal soportada en PyTorch ≥2.3.**
  Pasar `attn_mask=mask` y `is_causal=True` al mismo `nn.MultiheadAttention` emite warning silencioso y deshabilita el fast-path FlashAttention; en futuras versiones el comportamiento puede pasar a indefinido.
  **Fix aplicado:** `is_causal=False` + `attn_mask=mask` explícito. Cubierto por `test_tft_fusion_causal_mask_shape` y `test_tft_fusion_is_causal_under_future_perturbation`.

- [x] **2. `tft_attention.py:136` — `attn_weights = torch.empty(0)` rompe contrato de shape.**
  Cuando `MultiheadAttention` devolvía `None` (FlashAttention / `need_weights=False`), se sustituía por tensor 1D vacío. Consumidores aguas abajo (`ensemble.py:74,82`) hacían `attn_weights.cpu().numpy()` esperando `(batch, seq, seq)`.
  **Fix aplicado:** se devuelve `torch.zeros(batch, [num_heads,] seq, seq, device=x.device, dtype=x.dtype)` con la shape correcta según `average_attn_weights`.

- [x] **3. `calibration.py:94-109` — Race condition en `update_in_background`.**
  `self._updating` se seteaba **fuera** del lock y se asignaba **dentro** del thread → dos invocaciones lanzaban dos threads que reentrenaban `IsotonicRegression` en paralelo.
  **Fix aplicado:** la guarda `_update_in_progress` se chequea y se setea bajo el mismo `self._lock` antes del `Thread.start()`. Cubierto por `test_calibrator_update_in_background_is_race_free` (8 threads concurrentes).

- [x] **4. `calibration.py:115-119` — Lectura no atómica del par `(x_thresholds, y_values)`.**
  Antes se asignaban dos campos separados bajo lock pero se leían sin lock → tuple-tearing entre `x` nuevo e `y` viejo.
  **Fix aplicado:** snapshot único `self._curve = (x_th, y_th)` como tupla inmutable; `calibrate_signal` la lee como una sola referencia atómica (GIL la garantiza para una asignación).

- [x] **5. `meta_learner.py:83-95, 109-111` — API SHAP obsoleta para multi-class.**
  Asumía `list[ndarray]` por clase; SHAP ≥0.42 devuelve `ndarray (n, f, k)`.
  **Fix aplicado:** helper `_normalize_shap_values` que maneja ambos formatos (lista legacy, 2D binaria y 3D moderna). Cubierto por `test_meta_learner_shap_modern_api`.

- [x] **6. `ensemble.py:26-90` + `hybrid_tft.py:24-84` — Duplicación arquitectónica.**
  Dos pipelines CNN→LSTM→TFT→GRN→Linear coexistían sin compartir definición.
  **Fix aplicado:** `HybridSignalEngine` ahora compone un `HybridCNNLSTMTFT` interno como backbone único; sólo añade calibrador + meta-learner + cabezal binario + I/O.

---

## P2 — Altos (contratos rotos, parámetros muertos, integraciones frágiles)

- [x] **7. `hybrid_tft.py:33` — `cnn_channels` aceptado y nunca usado.**
  El extractor tenía canales hardcoded `(64, 128)`.
  **Fix aplicado:** `cnn_channels` se propaga a `CNN1DExtractor(channels=...)`. Acepta `int` (auto-expande a `(c, c*2)`) o secuencia.

- [x] **8. `meta_learner.py:23, 31` — `n_splits` aceptado y nunca usado.**
  **Fix aplicado:** método `cross_val_mlogloss()` que ejecuta `TimeSeriesSplit(n_splits=...)` y reporta mlogloss por fold. Cubierto por `test_meta_learner_cross_val_mlogloss_shape`.

- [x] **9. `ensemble.py:93-96` — `generate_signal` no movía `x_window` al device del modelo.**
  **Fix aplicado:** helper `_device()` + `extract_features` mueve la entrada con `non_blocking=True`. Cubierto por `test_hybrid_signal_engine_extract_features_device_aware`.

- [x] **10. `ensemble.py:74-83` — `.cpu().numpy()` en cada inferencia bloqueaba sincronía CUDA.**
  **Fix aplicado:** `extract_features` devuelve tensores en device por default; sólo convierte a NumPy si `as_numpy=True` (path para meta-learner XGBoost).

- [x] **11. `tft_attention.py:138-141` — Doble residual / doble LayerNorm.**
  La `GRN` ya hace `LayerNorm(x + residual)`; envolver con otro `norm2(x + grn(x))` desperdiciaba parámetros y rompía la interpretabilidad del bloque.
  **Fix aplicado:** se asigna directamente `x = self.grn(x)` tras la atención.

- [x] **12. `tft_attention.py:130-134` — `need_weights` implícito.**
  **Fix aplicado:** se declara `need_weights=True, average_attn_weights=self.average_attn_weights` explícito; parámetro `average_attn_weights` expuesto en el constructor.

- [x] **13. `cnn_extractor.py:5-20` — `CausalConv1d` no validaba `stride>1` ni `groups>1`.**
  **Fix aplicado:** el constructor sólo permite `stride=1` (forzado en `super().__init__`) y valida `kernel_size>=1` + `dilation>=1`. Cubierto por `test_causal_conv1d_rejects_stride_gt_1`. **Bonus**: descubrimos que `GroupNorm` también rompía causalidad (mezcla stats por el eje temporal) — reemplazado por `ChannelLayerNorm`. Cubierto por `test_cnn_extractor_is_causal_under_future_perturbation`.

- [x] **14. `meta_learner.py:47-55` — Sin validación real de etiquetas ni soporte de desbalanceo.**
  **Fix aplicado:** `_validate_labels` exige `set(y) ⊆ {0,1,2}`; `class_weight='balanced'` o `dict[int,float]`, además de `sample_weight` explícito. Cubierto por `test_meta_learner_rejects_out_of_range_labels` y `test_meta_learner_class_weight_balanced`.

- [x] **15. `meta_learner.py:13` — `warnings.filterwarnings(...)` global a nivel de módulo.**
  **Fix aplicado:** se encierra en `warnings.catch_warnings()` local alrededor del `fit`.

- [x] **16. `ensemble.py:64-77` — `extract_features` cambiaba `self.training` cada llamada.**
  **Fix aplicado:** `extract_features` ya no toca el modo; usa `@torch.inference_mode()`. El caller gestiona `eval()`.

---

## P3 — Medios (calidad / consistencia)

- [x] **17. `hybrid_tft.py:75` — Docstring incorrecto** (decía `(logits[:, 1], attn_weights)` pero `Linear(_, 1)` produce un único logit).
  **Fix aplicado:** docstring rescrito; `forward` documenta el shape `(B, 1)`.

- [x] **18. `ensemble.py:90, 92` — `feature_names` recibido y no usado.**
  **Fix aplicado:** parámetro eliminado del `generate_signal`. La lista de features vive en el `FeatureBuilder` (`src/data/features.py`) y se expone via `dataset.feature_names`. La explicación SHAP se delega al `meta_learner.get_regime_explanation(X, feature_names=...)`.

- [x] **19. `ensemble.py:106, 109-119` — Lógica de negocio hardcoded.**
  Umbrales `0.70/0.30/0.80/0.20`, sizing `1.5/0.8/1.0/0.4`, etiquetas `["LOW_VOL","TRENDING","HIGH_VOL"]` y mapeo de ruta a régimen.
  **Fix aplicado:** `SignalPolicy` dataclass parametriza umbrales/sizing/ruta; `regime_labels` viene del meta-learner. Cubierto por `test_signal_policy_rejects_inconsistent_thresholds`.

- [x] **20. `bilstm_encoder.py:55` — `rnn_type` no validado.**
  **Fix aplicado:** `_VALID_RNN_TYPES = {"lstm","gru"}`, raise si no está. Cubierto por `test_bilstm_rejects_invalid_rnn_type`.

- [x] **21. `bilstm_encoder.py:81-86` y `cnn_extractor.py:60-65` — `return_sequence=True` sin Dropout.**
  **Fix aplicado:** ambas ramas (colapsada y secuencial) aplican el mismo Dropout configurado.

- [x] **22. `cnn_extractor.py:41, 46` — `GroupNorm(num_groups=8)` mágico.**
  **Fix aplicado:** `group_norm_groups` paramétrico con auto-derivado `gcd(channels, requested)`. Adicionalmente, **reemplazamos GroupNorm por ChannelLayerNorm** (P1 oculto: GroupNorm rompía causalidad). Cubierto por `test_cnn_extractor_auto_groups_for_odd_channels`.

- [x] **23. `tft_attention.py:127` — Máscara causal recomputada en cada forward.**
  **Fix aplicado:** máscara `triu` se pre-allocan en `register_buffer` (`_causal_mask_cache`) y se trunca al `seq_len` real. Fallback on-the-fly si la secuencia excede el cache.

- [x] **24. `meta_learner.py:72-95, 97-122` — `get_regime_explanation` y `get_shap_explanations` duplican código.**
  **Fix aplicado:** helper privado `_shap_top_features` consume ambos; cada API pública sólo decide qué `regime_indices` pasar.

- [x] **25. `calibration.py:75` — Umbral `< 100` hardcoded.**
  **Fix aplicado:** parámetro `min_observations` (default 100) en el constructor del calibrador.

- [x] **26. `bilstm_encoder.py:104-110` — Type hints no distinguen 2D/3D según `return_sequence`.**
  **Fix aplicado:** `@overload` agregado a `forward` (returns `torch.Tensor` con la documentación de shape variando por flag).

- [x] **27. `ensemble.py:122` — Timestamp del signal usa `datetime.now()` en vez del epoch de mercado.**
  **Fix aplicado:** parámetro `as_of_epoch: Optional[int]` en `generate_signal`; cae a `datetime.now(timezone.utc)` sólo si es `None`. Cubierto por `test_hybrid_signal_engine_generate_signal_returns_full_payload`.

- [x] **28. `calibration.py` — Sin métricas de calidad expuestas.**
  **Fix aplicado:** métodos `brier_score()` y `expected_calibration_error(n_bins=...)`. Cubierto por `test_calibrator_brier_and_ece_bounded`.

- [x] **29. `cnn_extractor.py` — Sin test de causalidad explícito.**
  **Fix aplicado:** `test_cnn_extractor_is_causal_under_future_perturbation` perturba `x[:, 8:]` en `+100` y exige invariancia en `y[:, :8]` (a tolerancia 1e-5). Mismo test añadido para `TFTFusionNode`.

- [x] **30. `meta_learner.py` — No expone `feature_importances_` ni hook de hyper-tuning.**
  **Fix aplicado:** `@property feature_importances_` que delega a `model.feature_importances_`. Cubierto por `test_meta_learner_feature_importances_exposed`. *Hyper-tuning: aún no expuesto (ver P4).*

---

## P4 — Bajos (pulido y rendimiento)

- [x] **31. `bilstm_encoder.py` — `embedding_dim` por defecto coincide con `hidden_size` (64) por accidente.**
  **Fix aplicado:** `embedding_dim` ahora es `Optional[int]`; si es `None`, se deriva como `hidden_size * (2 if bidirectional else 1)`. Docstring documenta explícitamente la separación. Cubierto por `test_bilstm_embedding_dim_defaults_to_hidden_size_for_unidirectional`, `_2x_for_bidirectional` y `_explicit_override`.

- [x] **32. Toda la familia — Sin soporte `torch.amp.autocast` ni `.half()` coherente.**
  **Fix aplicado:** `Trainer.precision={fp32,fp16,bf16}` ya soportaba training. Añadidos tests específicos de **inferencia**: `test_hybrid_signal_engine_fp16_inference_matches_fp32_within_tolerance` (con `.half()`, tol 1e-2) y `test_hybrid_signal_engine_bf16_autocast_inference` (con `torch.autocast` BF16, tol 2e-2).

- [x] **33. `calibration.py` — `deque → np.array` cada update es O(N).**
  **Fix aplicado:** `deque` reemplazado por dos `np.ndarray` pre-allocados (`_margins_buf`/`_labels_buf`) de tamaño `window_size` + cursor `_head` + contador `_count`. `_snapshot_buffer()` retorna vista contigua (sin wrap-around) o concatenación de dos vistas (con wrap-around). El test existente `test_calibrator_monotonic_after_fit` sigue pasando — semántica preservada.

- [x] **34. Toda la familia — Cada archivo lleva su propio `__main__` con prints/seeds.**
  **Fix aplicado:** el último `__main__` con prints (`src/features/generator.py`) fue eliminado; la versión equivalente vive en `scripts/verify_generator.py`. `grep -rn "if __name__" src/` ahora retorna 0 hits dentro de `src/`.

- [x] **35. `tft_attention.py` — Sin máscara de padding.**
  **Fix aplicado:** `key_padding_mask: Optional[torch.Tensor]` ya estaba expuesto; añadido `test_tft_fusion_key_padding_mask_zeros_out_padded_attention` que verifica que (a) los outputs en las posiciones válidas son **bit-exactos** a una versión sin padding del mismo largo válido, y (b) los pesos de atención hacia las posiciones paddeadas son ~0.

---

## Trabajo nuevo más allá de la auditoría (entregado en `a270600`)

- [x] `src/models/heads.py` — `MultiContractMultiHorizonHead` (CALL/PUT, HIGHER/LOWER, TOUCH/NOTOUCH, ENDSIN/OUT, DIGITEVENODD × horizons).
- [x] `src/models/conditioning.py` — `AssetTimeframeEmbedding` con vocab dinámico (cualquier símbolo Deriv × {ticks + 60s…86400s}).
- [x] `src/data/` completo — `store_adapter`, `features` (switch ticks vs candles), `labels` (Deriv-aware con `IGNORE_LABEL`), `dataset` (`WindowDataset`), `sampler` (purged + DDP-aware).
- [x] `src/training/` completo — config dataclasses, `Trainer` (auto-detect CPU/single-GPU/DDP), `MultiContractLoss`, AMP fp16/bf16, checkpoints, early stopping. **DDP smoke test real** con backend `gloo` (2 ranks).
- [x] `tests/conftest.py` — auto-skip de tests `pandas-ta`-dependientes cuando falta la dependencia (Python 3.11).

---

## Lo que aún falta (roadmap inmediato)

### Capa de aplicación / serving
- [x] **A1. Live inference loop** *(entregado en `5640462`)* — `scripts/infer.py` consume ticks de `DerivWebSocketClient.ticks_history_stream`, mantiene ventana rodante con el `FeatureBuilder` correcto, calibra con `PerContractCalibratorBundle`, aplica `SignalPolicy` y emite JSON por stdout. Soporta `--max-iterations` para CI.
- [x] **A2. Backtester walk-forward** — `src/backtest/` con `engine.py` (event-driven simulator binario con payout/commission/sizing), `metrics.py` (Sharpe, Sortino, max drawdown + duration, profit factor, win rate, breakdown por contrato + annualization factor), `walk_forward.py` (orquestrador expanding/rolling con purga y embargo entre train/val/test). CLI `scripts/backtest.py` con modos `walk-forward` y `static`. 18 tests cubriendo métricas analíticas, engine determinístico (CALL/PUT/NO_TRADE/masked), aritmética de PnL exacta, walk-forward ranges y CLI smoke end-to-end.
- [x] **A3. Risk manager** — `src/risk/manager.py` con `RiskConfig` declarativo (max_drawdown, max_daily_loss, max_trades_per_day, max_trades_per_contract, max_concurrent_exposure), `RiskState` mutable (cumulative PnL, peak, drawdown, daily counters, open exposure, kill-switch), `RiskManager` thread-safe con `evaluate(...)→RiskDecision(allow, adjusted_sizing, reason)` pre-trade y `record_trade(...)` post-trade. Integrado en `BacktestEngine.risk_manager` (downgradea a NO_TRADE o ajusta sizing) y expuesto via `scripts/backtest.py` con flags `--max-drawdown`, `--max-daily-loss`, `--max-trades-per-day`, `--max-concurrent-exposure`. 13 tests: drawdown kill-switch, daily loss + reset por día, trades-per-day cap, per-contract cap, exposure cap con reducción de sizing, release_exposure, integración end-to-end con BacktestEngine (CALL forzado sobre serie decreciente → kill-switch tras N pérdidas → resto NO_TRADE).

### Calibración / post-procesado
- [x] **B1.** Calibrador **por contrato** *(entregado en `5640462`)* — `PerContractCalibratorBundle` mantiene una `LowLatencyRollingIsotonicCalibrator` por `(contract, horizon)` con add/calibrate vectorizado, `state_dict` round-trip y `quality_report()` (Brier + ECE por celda).
- [x] **B2. Re-calibración online auto-trigger por drift de Brier/ECE.** `src/models/drift.py`: `OnlineCalibrationMonitor` con umbrales `max_brier`, `max_ece`, `min_observations`, **hysteresis** vía `recovery_margin` (una celda en alerta vuelve a OK sólo si su métrica cae a/por debajo de `threshold - recovery_margin`) y **cooldown** entre refits (`cooldown_seconds` medido con el `epoch` que pasa el caller para mantener determinismo). API: `check(bundle, now_epoch) → dict[cell, DriftDecision]` y `maybe_refit(bundle, decisions, now_epoch, background=True) → int` que dispara `update_in_background`/`update_calibration_curve` sólo en las celdas marcadas y respeta el cooldown. 11 tests: validaciones, `insufficient_observations`, `ok` para bundle bien calibrado, refit en bundle mal calibrado, cooldown bloquea back-to-back, cooldown expira tras el threshold, hysteresis con tolerancia exacta a métricas en cero, refit en background con `wait_update_done`, reset.
- [x] **B3. Conformal prediction encima del cabezal CALL/PUT.** `src/models/conformal.py`: `InductiveConformalPredictor` (ICP binario con score `1 - p̂(y|x)`, ring buffer adaptativo, quantile cacheado) + `ConformalBundle` paralelo al calibrador con API vectorizada `(B,C,H)→(B,C,H,2)`. Integrado en `BacktestEngine.conformal_gate`: cuando el set conformal no es `{0}` ni `{1}`, la celda se fuerza a NO_TRADE (preserva la garantía marginal de coverage ≥1-α). 14 tests cubriendo: properties del set (singleton/ambivalent/empty), validación, coverage empírico ≥ 1-α en datos sintéticos, monotonicidad respecto a α, ring buffer wrap-around, bundle vectorizado y dos escenarios end-to-end con backtester (gate ambivalente forcing NO_TRADE vs gate calibrado dejando pasar señales).

### Entrenamiento end-to-end
- [ ] **C1.** Script `scripts/train.py` que: carga `TrainingConfig` desde YAML/JSON → instancia `WindowDataset` (multi-symbol vía `ConcatDataset`) → backbone + cabezales + embedding → `Trainer.fit()` → escribe el mejor checkpoint a disco.
- [ ] **C2.** Spawn DDP real (`torch.multiprocessing.spawn`) wrappeado en `scripts/train.py --world-size N`.
- [ ] **C3.** Reanudación de entrenamiento (`--resume <ckpt>`).

### Datos
- [ ] **D1.** Adaptador a `src/features/generator.py` cuando `pandas-ta` esté disponible (fallback automático al `CandleFeatureBuilder`).
- [ ] **D2.** Multi-symbol `WindowDataset` (concatenación + `symbol_id` correcto por sample).
- [ ] **D3.** Caché de features pre-computadas en Parquet (rehacer ventanas es lento si rasgamos features cada batch).

### Observabilidad
- [x] **E1. Logging estructurado JSON con `correlation_id` por señal.** `src/observability/logging.py`: `JsonFormatter` que serializa cada `LogRecord` a una línea JSON estable (`ts`, `level`, `logger`, `message`, `correlation_id`, extras vía `extra={...}`, `exception`/`stack` cuando aplica); `correlation_id` como `contextvars.ContextVar` thread-safe + asyncio-safe (Python copia el contexto en task_factory por default); `configure_root(level, stream, json_format)` idempotente. 7 tests.
- [x] **E2. Métricas Prometheus.** `src/observability/metrics.py`: wrapper import-safe sobre `prometheus_client`. `MetricsRegistry` opinado con `counter/gauge/histogram` factory de-duplicado por nombre; **si `prometheus_client` no está, todo degrada a `_NoOpMetric`** silenciosa sin warnings. Métricas estándar pre-instanciadas: `inference_latency_seconds` (buckets pensados para target p99 < 5ms), `train_batch_duration_seconds`, `train_loss` (labels stage+contract), `signals_emitted_total` (signal+contract). `start_http_server(port)` opcional. 5 tests (incluye fallback path simulado con monkeypatch).
- [ ] **E3.** Trazas OpenTelemetry desde `generate_signal`.

### CI / dev experience
- [x] **F1. `pyproject.toml` extras.** Nuevos grupos opcionales:
  - `training` — torch + duckdb
  - `tuning` — optuna ≥3.6
  - `backtest` — torch + duckdb
  - `serving` — torch + websockets + httpx
  - `legacy-features` — pandas-ta (movido fuera del core porque sólo tiene wheels para Py 3.12+)
  - `full` — meta-extra que une los anteriores
  - `dev` ampliado con torch, duckdb, optuna, websockets
  - `requires-python` bajado de `>=3.12` a `>=3.11` (con la salvedad documentada)
  - `[tool.hatch.build.targets.wheel].packages` ampliado con `src/data`, `src/training`, `src/backtest`, `src/risk`
- [x] **F2. GitHub Actions con matrix Python 3.11 / 3.12.** `.github/workflows/test.yml` con:
  - Job `pytest` matrix:
    - **3.11**: extras `dev,deriv-ingest,training,tuning,backtest,serving` (sin pandas-ta; `conftest.py` auto-skipea los tests legacy).
    - **3.12**: extras anteriores + `legacy-features,analysis` (suite completa incl. `test_feature_generator` + `test_integrity`).
  - Verificación de imports antes de correr tests (catch errores de empaquetado).
  - `pytest -W error` para tratar warnings como errores.
  - Upload de pytest cache en failure.
  - Job `lint` con `ruff==0.1.15` sobre `src/ tests/ scripts/`.
  - `concurrency` cancela jobs viejos al pushear.
- [x] **F3. Pre-commit hooks.** `.pre-commit-config.yaml` con:
  - `pre-commit-hooks` v4.6.0: trailing-whitespace, end-of-file-fixer, check-merge-conflict, check-yaml, check-toml, check-json, check-added-large-files (cap 2 MB), detect-private-key.
  - `ruff` v0.1.15 (mismo pin que `[dev]`) con `--fix --exit-non-zero-on-fix`.
  - `mypy` v1.8.0 sobre `src/` con `--check-untyped-defs --ignore-missing-imports`; dependencies adicionales numpy<2, pandas 2.2, pydantic 2.6 inyectadas para que el type-check no falle por imports faltantes.
  - Bloque `ci:` con mensajes de commit estandarizados (`autofix_commit_msg` y `autoupdate_commit_msg`).
- [x] **F4. Benchmarks de latencia.** `scripts/benchmark_inference.py`:
  - Mide latencia end-to-end del pipeline `(W, F) → BackboneWithHeads → calibrate → SignalPolicy` por iteración con `torch.inference_mode` y `cuda.synchronize` cuando aplica.
  - Reporta p50/p95/p99/mean/stdev/min/max en milisegundos via `BenchmarkReport` dataclass.
  - Modos `--mode forward|full` (con/sin calibrator + policy).
  - Target **p99 < 5ms** documentado; `--target-p99-ms` parametrizable; `--fail-on-regression` retorna exit code 1 si supera el target (gating para CI).
  - `--json` para integración con dashboards de regresión.
  - Warmup de 30-50 iteraciones para excluir JIT/cuDNN tuning.
  5 tests smoke: shapes del reporte, modos forward vs full, validaciones, CLI JSON output, exit-code-1 con target imposible.

### Hyperparameter tuning (Optuna)

- [x] **G1. Wrapper `src/training/tuning.py`** — `BackboneObjective` + `tune(study, n_trials, ...)`. `SearchSpace` dataclass declarativo (`FloatRange/IntRange/Categorical`); cada campo `None` queda fijo a la `base_config`. Soporta tuples categóricos (e.g. `cnn_channels`) via repr/eval round-trip.
- [x] **G2. Métrica objetivo: Brier post-calibración** — `target="brier"` calibra un `PerContractCalibratorBundle` sobre val tras cada fold y devuelve el Brier promedio por celda. Alternativa `target="val_loss"` para fallback rápido.
- [x] **G3. Pruner `MedianPruner(n_warmup_steps=2, n_min_trials=5)`** — default en `tune()`. `trial.report` por fold + `should_prune` corta trials malos temprano.
- [x] **G4. Walk-forward dentro del trial** — `k_folds` expanding-window con `purge` derivado de `max(horizons)`. Promedio de las métricas de los folds → métrica final del trial.
- [x] **G5. Storage SQLite** — `tune(..., storage="sqlite:///optuna_studies/<study>.db")` con `load_if_exists`. Cubierto por `test_tune_sqlite_storage_round_trip`.
- [x] **G6. Estrategia DDP/trial** — cada trial corre single-GPU/CPU (no DDP) por diseño del `Trainer`; Optuna paraleliza con `n_jobs=k` expuesto en el CLI.
- [x] **G7. Tuner XGBoost** — `XGBoostMetaLearnerObjective` con `early_stopping_rounds` + reporte por fold para que `MedianPruner` corte trials malos. Tunea `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`.
- [x] **G8. CLI `scripts/tune.py`** — `--target {backbone,xgboost}`, `--study-name`, `--n-trials`, `--storage`, `--n-jobs`. Para backbone reusa los flags de `scripts/train.py` (`--db`, `--symbol`, `--window-size`, `--horizons`, `--contracts`). Para xgboost toma `--xgb-X`/`--xgb-y` desde `.npy`/`.npz`.
- [x] **G9. Tests** — 9 nuevos con `RandomSampler` + `n_trials=2` + `k_folds=2` + `max_epochs=1`. Cubren: sampling del `SearchSpace`, ambos targets (brier/val_loss), prune por dataset chico, prune via MedianPruner, persistencia SQLite con resume, y smoke de ambos modos del CLI.
