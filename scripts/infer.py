"""Inferencia en vivo: WebSocket Deriv → ventana rodante → señal calibrada.

Loop:

1. Conecta al ``DerivWebSocketClient`` y suscribe a ``ticks_history_stream``
   (puede usar style="candles" con granularity, o style="ticks").
2. Acumula los últimos ``window_size`` valores OHLC / quote en un buffer.
3. Construye features causales con el mismo ``FeatureBuilder`` que usó el
   entrenamiento (ticks vs candles según el stream).
4. Cuando hay ventana llena, ejecuta ``BackboneWithHeads.forward`` y
   pasa los logits por ``PerContractCalibratorBundle.calibrate``.
5. Aplica la ``SignalPolicy`` y emite por stdout (o webhook) un
   payload JSON con la señal.

Diseño:

* Cero magia de estado: todas las piezas (modelo, calibrador, embedding,
  feature builder) se levantan a partir del checkpoint y del config del
  entrenamiento.
* El loop es interruptible (Ctrl-C) y cierra el WebSocket cleanly.
* Si el modelo está en CPU, todo corre en CPU; si hay CUDA, se mueve a
  ``cuda:0`` automáticamente.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.connectors.deriv.client import DerivWebSocketClient
from src.data.features import (
    CandleFeatureBuilder,
    FeatureBuilderConfig,
    TickFeatureBuilder,
    build_feature_builder,
)
from src.models.calibration_bundle import PerContractCalibratorBundle
from src.models.composite import BackboneWithHeads, build_model_from_config
from src.models.conditioning import AssetTimeframeEmbedding
from src.models.ensemble import SignalPolicy
from src.models.heads import HeadConfig
from src.training.config import ModelConfig

log = logging.getLogger("infer")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True, help="ruta al best.pt")
    p.add_argument(
        "--calibrator-bundle", type=str, default=None,
        help="JSON con curvas isotónicas por contrato",
    )
    p.add_argument("--app-id", type=str, required=True, help="Deriv app_id")
    p.add_argument(
        "--endpoint", type=str, default="wss://ws.derivws.com/websockets/v3"
    )
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument(
        "--style", type=str, choices=("ticks", "candles"), default="candles"
    )
    p.add_argument("--granularity", type=int, default=60)
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument(
        "--contracts", type=str, nargs="+",
        default=["CALLPUT", "HIGHERLOWER"],
    )

    # Modelo (debe coincidir con el del checkpoint).
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--lstm-hidden", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--cnn-channels", type=int, nargs="+", default=[64, 128])
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument(
        "--device", type=str, choices=("auto", "cpu", "cuda"), default="auto"
    )
    p.add_argument("--max-iterations", type=int, default=None,
                   help="parar tras N inferencias (útil para smoke / CI)")
    p.add_argument("--log-level", type=str, default="INFO")
    return p


# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------


def _pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; using CPU")
            return torch.device("cpu")
        return torch.device("cuda:0")
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def load_model_and_bundle(
    args: argparse.Namespace,
    num_features: int,
) -> tuple[BackboneWithHeads, PerContractCalibratorBundle, torch.device]:
    device = _pick_device(args.device)
    embedding = AssetTimeframeEmbedding(embedding_dim=32)
    embedding.register_symbol(args.symbol)
    embedding.register_granularity(args.granularity if args.style == "candles" else None)

    head = HeadConfig(
        contracts=tuple(args.contracts), horizons=tuple(args.horizons),
        use_context=True, dropout=args.dropout,
    )
    cfg = ModelConfig(
        embedding_dim=args.embedding_dim,
        lstm_hidden=args.lstm_hidden,
        num_attention_heads=args.num_heads,
        cnn_channels=tuple(args.cnn_channels),
        dropout=args.dropout,
        head=head,
    )
    model = build_model_from_config(
        cfg,
        num_features=num_features,
        sequence_length=args.window_size,
        embedding=embedding,
    ).to(device)

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    log.info("loaded model from %s (%.2fM params)",
             args.checkpoint, model.count_parameters() / 1e6)

    bundle = PerContractCalibratorBundle(
        contracts=args.contracts, horizons=args.horizons,
    )
    if args.calibrator_bundle:
        raw = json.loads(Path(args.calibrator_bundle).read_text())
        state = {
            k: {
                "x_thresholds": np.asarray(v["x_thresholds"], dtype=np.float64),
                "y_values": np.asarray(v["y_values"], dtype=np.float64),
            }
            for k, v in raw.items()
        }
        bundle.load_state_dict(state)
        log.info("loaded calibrator bundle (%d cells)", len(state))
    return model, bundle, device


# ---------------------------------------------------------------------------
# Live loop
# ---------------------------------------------------------------------------


def _feature_builder_for(style: str) -> "object":
    return build_feature_builder(
        kind="ticks" if style == "ticks" else "candles",
        config=FeatureBuilderConfig(),
    )


def _row_from_message(style: str, msg: dict) -> Optional[dict]:
    """Extrae una fila ``DataFrame``-able del payload Deriv."""
    if style == "ticks":
        tick = msg.get("tick")
        if tick is None:
            return None
        return {
            "epoch": int(tick["epoch"]),
            "quote": float(tick["quote"]),
            "bid": float(tick.get("bid", tick["quote"])),
            "ask": float(tick.get("ask", tick["quote"])),
        }
    ohlc = msg.get("ohlc")
    if ohlc is None:
        return None
    return {
        "epoch": int(ohlc["epoch"]),
        "open": float(ohlc["open"]),
        "high": float(ohlc["high"]),
        "low": float(ohlc["low"]),
        "close": float(ohlc["close"]),
    }


async def run_inference(args: argparse.Namespace) -> None:
    fb = _feature_builder_for(args.style)
    # Pre-fit con una ventana fake para que ``num_features`` sea conocido.
    if isinstance(fb, CandleFeatureBuilder):
        warm = pd.DataFrame({
            "epoch": np.arange(args.window_size + 30),
            "open": np.linspace(1, 2, args.window_size + 30),
            "high": np.linspace(1.1, 2.1, args.window_size + 30),
            "low": np.linspace(0.9, 1.9, args.window_size + 30),
            "close": np.linspace(1, 2, args.window_size + 30),
        })
    else:
        warm = pd.DataFrame({
            "epoch": np.arange(args.window_size + 30),
            "quote": np.linspace(1, 2, args.window_size + 30),
            "bid": np.linspace(1, 2, args.window_size + 30),
            "ask": np.linspace(1, 2, args.window_size + 30),
        })
    _ = fb.fit_transform(warm)
    num_features = fb.num_features
    log.info("feature builder ready: %d features", num_features)

    model, bundle, device = load_model_and_bundle(args, num_features=num_features)
    policy = SignalPolicy()

    buffer: deque[dict] = deque(maxlen=args.window_size + max(args.horizons) + 10)
    iterations = 0

    async with DerivWebSocketClient(app_id=args.app_id, endpoint=args.endpoint) as ws:
        stream = ws.ticks_history_stream(
            args.symbol,
            style=args.style,
            granularity=args.granularity if args.style == "candles" else None,
            count=args.window_size + 30,
            end="latest",
        )
        async for msg in stream:
            row = _row_from_message(args.style, msg)
            if row is None:
                continue
            buffer.append(row)
            if len(buffer) < args.window_size:
                continue

            df = pd.DataFrame(list(buffer))
            features = fb.fit_transform(df)  # causal → idempotente al final
            window = features[-args.window_size :]
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            sym_id = torch.tensor(
                [model.context.symbol_id(args.symbol)], dtype=torch.long, device=device
            )
            gran_id = torch.tensor(
                [model.context.granularity_id(
                    args.granularity if args.style == "candles" else None
                )],
                dtype=torch.long, device=device,
            )

            with torch.inference_mode():
                logits = model(x, sym_id, gran_id)  # (1, C, H)
            probs = bundle.calibrate(logits)  # (1, C, H)

            payload = _build_signal_payload(
                args, df.iloc[-1]["epoch"], logits[0], probs[0], policy,
            )
            print(json.dumps(payload), flush=True)

            iterations += 1
            if args.max_iterations is not None and iterations >= args.max_iterations:
                break


def _build_signal_payload(
    args: argparse.Namespace,
    epoch: int,
    logits: torch.Tensor,
    probs: np.ndarray,
    policy: SignalPolicy,
) -> dict:
    out: dict = {
        "epoch": int(epoch),
        "symbol": args.symbol,
        "style": args.style,
        "granularity": args.granularity if args.style == "candles" else None,
        "predictions": {},
    }
    for ci, contract in enumerate(args.contracts):
        for hi, horizon in enumerate(args.horizons):
            p = float(probs[ci, hi])
            signal, sizing = _classify(p, policy)
            key = f"{contract}__h{int(horizon)}"
            out["predictions"][key] = {
                "logit": float(logits[ci, hi].item()),
                "p_calibrated": round(p, 6),
                "signal": signal,
                "sizing_multiplier": sizing,
            }
    return out


def _classify(p: float, policy: SignalPolicy) -> tuple[str, float]:
    if p >= policy.call_threshold:
        return ("CALL", policy.strong_sizing if p >= policy.strong_call_threshold else policy.normal_sizing)
    if p <= policy.put_threshold:
        return ("PUT", policy.strong_sizing if p <= policy.strong_put_threshold else policy.normal_sizing)
    return ("NO_TRADE", policy.no_trade_sizing)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run_inference(args))
    except KeyboardInterrupt:
        log.info("interrupted by user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
