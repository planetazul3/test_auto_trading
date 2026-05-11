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

## P4 — Bajos *(pantalla no adjunta — placeholder para completar)*

> Pendiente recibir el detalle exacto de la captura faltante. Items típicos esperados (sin confirmar):
- [ ] **31.** Docstrings de retorno faltantes en `hybrid_tft.forward` / `tft_attention.forward`.
- [ ] **32.** Sin hook de hyper-tuning expuesto en `meta_learner` (Optuna/Hyperopt).
- [ ] **33.** Init de pesos no documentado en backbone (`xavier/orthogonal/forget-bias=1`).
- [ ] **34.** `repr()` / `extra_repr()` no informativo en módulos custom.
- [ ] **35.** Falta `torch.jit.script`/`torch.compile` smoke test.
- [ ] **36.** No hay `model.summary()` / count de parámetros expuesto.
- [ ] **37.** Falta logging estructurado en cambios de régimen.
- [ ] **38.** Sin contract test del payload JSON de `generate_signal` (schema fijo).

> **Acción requerida**: enviar foto/texto de P4 para reemplazar este placeholder por la lista exacta y trackearla con el mismo formato.

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
- [ ] **A1. Live inference loop**: consumir ticks desde `DerivWebSocketConnector`, mantener ventana rodante, llamar `engine.generate_signal()` en streaming y publicar señales (Redis Stream o socket).
- [ ] **A2. Backtester walk-forward**: simular fills usando históricos del DuckDB, evaluar PnL por contrato y por régimen.
- [ ] **A3. Risk manager**: límite por contrato/símbolo/régimen, kill-switch por drawdown, exposure cap.

### Calibración / post-procesado
- [ ] **B1.** Calibrador **por contrato** (no único): un `LowLatencyRollingIsotonicCalibrator` por cada cabezal de `MultiContractMultiHorizonHead`.
- [ ] **B2.** Re-calibración online del modelo (auto-trigger por drift de Brier/ECE).
- [ ] **B3.** Conformal prediction encima del cabezal CALL/PUT para garantizar coverage.

### Entrenamiento end-to-end
- [ ] **C1.** Script `scripts/train.py` que: carga `TrainingConfig` desde YAML/JSON → instancia `WindowDataset` (multi-symbol vía `ConcatDataset`) → backbone + cabezales + embedding → `Trainer.fit()` → escribe el mejor checkpoint a disco.
- [ ] **C2.** Spawn DDP real (`torch.multiprocessing.spawn`) wrappeado en `scripts/train.py --world-size N`.
- [ ] **C3.** Reanudación de entrenamiento (`--resume <ckpt>`).

### Datos
- [ ] **D1.** Adaptador a `src/features/generator.py` cuando `pandas-ta` esté disponible (fallback automático al `CandleFeatureBuilder`).
- [ ] **D2.** Multi-symbol `WindowDataset` (concatenación + `symbol_id` correcto por sample).
- [ ] **D3.** Caché de features pre-computadas en Parquet (rehacer ventanas es lento si rasgamos features cada batch).

### Observabilidad
- [ ] **E1.** Métricas Prometheus desde el Trainer (loss por step, lr, mem, throughput).
- [ ] **E2.** Logging estructurado JSON con `correlation_id` por señal.
- [ ] **E3.** Trazas OpenTelemetry desde `generate_signal`.

### CI / dev experience
- [ ] **F1.** `pyproject.toml` extras: `[dev]`, `[training]`, `[serving]` con sus deps.
- [ ] **F2.** GitHub Actions con matrix Python 3.11/3.12 (3.12 sí tiene `pandas-ta`).
- [ ] **F3.** Pre-commit hooks (ruff + mypy strict en `src/`).
- [ ] **F4.** Benchmarks de latencia de `generate_signal` (target p99 < 5ms en CPU).
