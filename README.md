# ML-Signal-Engine

Sistema híbrido de aprendizaje automático para la generación de señales de trading (PUT/CALL) basado en una arquitectura Ensemble de última generación.

## Descripción

Este proyecto implementa una arquitectura robusta que combina aprendizaje profundo (Deep Learning) con modelos clásicos de Machine Learning para predecir movimientos del mercado financiero. El motor unifica la extracción de patrones locales mediante CNN-1D, la memoria secuencial de BiLSTM y la fusión de características mediante mecanismos de atención (TFT), culminando en un meta-aprendiz XGBoost con calibración de probabilidades.

## Arquitectura

- **CNN-1D Extractor**: Identifica patrones geométricos y micro-tendencias en ventanas de datos OHLCV.
- **BiLSTM Encoder**: Captura dependencias temporales de largo plazo y el contexto secuencial.
- **TFT Fusion Node**: Nodo central inspirado en *Temporal Fusion Transformers* que utiliza *Multi-Head Attention* para ponderar la importancia de las diferentes fuentes de información.
- **XGBoost Meta-Learner**: Genera la señal final con calibración isotónica/sigmoide para una gestión de riesgo precisa.
- **Regime Awareness**: Detección de estados de mercado (tendencia, rango, volátil) mediante GMM (Gaussian Mixture Models) con prevención de *data leakage*.

## Instalación

Este proyecto utiliza `uv` para la gestión de dependencias.

```bash
# Sincronizar el entorno virtual
uv sync
```

## Uso

Para ejecutar el pipeline completo de entrenamiento y validación Walk-Forward:

```bash
uv run python main.py
```

Para visualizar los resultados y métricas en MLflow:

```bash
uv run mlflow ui
```

## Auditoría

El proyecto incluye una herramienta para consolidar todo el código fuente en un único archivo de auditoría:

```bash
uv run python scripts/generate_audit_dump.py
```

## Requisitos

- Python >= 3.12
- PyTorch
- XGBoost
- pandas-ta
- MLflow
- scikit-learn
- SHAP
