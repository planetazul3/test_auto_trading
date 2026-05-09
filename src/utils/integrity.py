"""
Validaciones de integridad temporal para el pipeline de features.

Dos chequeos complementarios:

1. ``assert_no_target_leakage``: chequeo barato post-feature-engineering que
   verifica que el target no esté duplicado en las features ni desplazado
   trivialmente. Se ejecuta sobre el dataframe ya etiquetado.

2. ``temporal_integrity_check``: el verdadero "Latent Leakage Test" del
   Blueprint §3.2. Ejecuta el pipeline sobre datos crudos, inyecta un spike
   puntual en el índice ``t_k`` y exige que las features para ``t < t_k`` no
   cambien respecto al baseline. Si cambian, el pipeline está propagando
   información futura hacia el pasado.
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd


def assert_no_target_leakage(
    df: pd.DataFrame,
    target_col: str = "target",
    horizon: int = 1,
    correlation_threshold: float = 0.99,
) -> bool:
    """
    Sanity-check estático: el target no debe estar entre las features ni
    aparecer desplazado por ``horizon`` pasos. Reporta correlaciones
    sospechosas pero no falla por ellas.
    """
    if target_col not in df.columns:
        raise ValueError(f"target '{target_col}' no existe en el dataframe.")

    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Correlaciones extremas (señal débil de duplicado encubierto). Suprimimos
    # warnings de columnas con varianza cero (correlación NaN benigna).
    numeric = features.select_dtypes(include=[np.number])
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = numeric.corrwith(target).dropna()
    suspicious = correlations[correlations.abs() > correlation_threshold]
    if not suspicious.empty:
        print(
            f"[integrity] correlaciones > {correlation_threshold} con el target:\n"
            f"{suspicious}"
        )

    # Detección de feature == target.shift(-horizon) exacto.
    shifted = target.shift(-horizon)
    for col in numeric.columns:
        if numeric[col].equals(shifted):
            raise ValueError(
                f"feature '{col}' es exactamente target.shift(-{horizon}); "
                "fuga directa del target."
            )

    return True


def temporal_integrity_check(
    feature_pipeline: Callable[[pd.DataFrame], pd.DataFrame],
    raw_df: pd.DataFrame,
    test_iterations: int = 5,
    spike_column: str = "close",
    spike_multiplier: float = 100.0,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    seed: Optional[int] = 1234,
) -> bool:
    """
    Latent Leakage Test (Blueprint §3.2).

    Estrategia:
      1. Computar baseline = pipeline(raw_df).
      2. Para cada iteración, inyectar un spike puntual en el índice ``t_k``
         de ``spike_column`` y volver a ejecutar el pipeline.
      3. Asertar que las features estrictamente anteriores a ``t_k`` son
         idénticas al baseline. Si difieren, el pipeline mezcla futuro en
         pasado.
    """
    if spike_column not in raw_df.columns:
        raise ValueError(f"columna '{spike_column}' no existe en raw_df.")

    rng = np.random.default_rng(seed)
    baseline = feature_pipeline(raw_df.copy())

    n = len(raw_df)
    lo = max(int(n * 0.2), 1)
    hi = max(int(n * 0.8), lo + 1)

    for it in range(test_iterations):
        t_k_pos = int(rng.integers(lo, hi))
        t_k_label = raw_df.index[t_k_pos]

        mutated_raw = raw_df.copy()
        col_idx = mutated_raw.columns.get_loc(spike_column)
        original_value = mutated_raw.iat[t_k_pos, col_idx]
        mutated_raw.iat[t_k_pos, col_idx] = original_value * spike_multiplier

        mutated = feature_pipeline(mutated_raw)

        # Trabajamos sobre el índice común. Después del dropna inicial de la
        # pipeline ambos dataframes comparten el mismo prefijo siempre que
        # la perturbación esté lejos del recorte; el filtro lo confirma.
        common_index = baseline.index.intersection(mutated.index)
        pre = common_index[common_index < t_k_label]
        if len(pre) == 0:
            continue

        baseline_pre = baseline.loc[pre]
        mutated_pre = mutated.loc[pre]

        try:
            pd.testing.assert_frame_equal(
                baseline_pre,
                mutated_pre,
                check_exact=False,
                rtol=rtol,
                atol=atol,
            )
        except AssertionError as exc:
            diffs = (baseline_pre - mutated_pre).abs().max()
            leaking = diffs[diffs > atol].sort_values(ascending=False)
            raise AssertionError(
                f"LATENT LEAKAGE: spike en {t_k_label} (pos={t_k_pos}, "
                f"iter={it}) alteró features previas. Columnas afectadas:\n"
                f"{leaking.head(10)}"
            ) from exc

    return True
