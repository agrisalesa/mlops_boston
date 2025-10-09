#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# xgboost es opcional; si no existe, el script continúa sin él
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
SCHEMA_PATH = MODELS / "schema.json"
METRICS_PATH = MODELS / "metrics.json"
META_PATH = MODELS / "model_metadata.json"
BEST_MODEL_PATH = MODELS / "best_model.pkl"
PREPROC_PATH = MODELS / "preprocessor.pkl"


def _load_processed() -> pd.DataFrame:
    path = PROCESSED / "housing_clean.csv"
    if not path.exists():
        raise FileNotFoundError("No existe data/processed/housing_clean.csv. Ejecuta 01_validar_datos.py primero.")
    return pd.read_csv(path)


def _schema_features_order(df: pd.DataFrame) -> list:
    """
    Ordena columnas de features.
    Si existe models/schema.json (en cualquiera de los dos formatos), usa ese orden.
    Si no, toma todas menos MEDV.
    """
    if SCHEMA_PATH.exists():
        sc = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        feats = sc.get("features", {})
        if isinstance(feats, dict):
            return list(feats.keys())
        if isinstance(feats, list):
            return feats
    return [c for c in df.columns if c != "MEDV"]


def _rmse(y_true, y_pred) -> float:
    # evita problemas con versiones antiguas de sklearn
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _guardar_schema_dict(feature_cols):
    """
    Guarda models/schema.json en el formato original solicitado:
      features como dict {col: {"dtype": "...", "binary": bool}}
    """
    def _infer_dtype(c: str) -> str:
        return "int" if c in ("CHAS", "RAD", "TAX") else "float"

    schema = {
        "target": "MEDV",
        "features": {
            c: {"dtype": _infer_dtype(c), "binary": bool(c == "CHAS")}
            for c in feature_cols
        },
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def main():
    MODELS.mkdir(parents=True, exist_ok=True)

    # 1) Cargar dataset procesado
    df = _load_processed()

    # 2) Definir columnas
    target = "MEDV"
    feature_cols = _schema_features_order(df)
    X = df[feature_cols].copy()
    y = df[target].copy()

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 4) Preprocesador simple (estandarización). Se guarda como artefacto.
    scaler = StandardScaler()
    xtr = scaler.fit_transform(X_train)
    xte = scaler.transform(X_test)

    # 5) Modelos
    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(xtr, y_train)
    models["linear_regression"] = lr

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,      # tamaño mínimo de hoja (controla sobreajuste)
        max_features="sqrt"      # número de features consideradas en cada split
    )
    rf.fit(xtr, y_train)
    models["random_forest"] = rf

    # Ridge
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(xtr, y_train)
    models["ridge"] = ridge

    if HAS_XGB:
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            reg_lambda=1.0,
        )
        xgb.fit(xtr, y_train)
        models["xgboost"] = xgb

    # 6) Evaluación
    metrics = {}
    for name, m in models.items():
        yhat = m.predict(xte)
        metrics[name] = _evaluate(y_test, yhat)

    # 7) Selección por RMSE mínimo
    best_name = min(metrics.keys(), key=lambda k: metrics[k]["rmse"])
    best_model = models[best_name]

    # 8) Guardado de artefactos
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    with open(PREPROC_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # 9) Guardar métricas y metadata
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    META_PATH.write_text(
        json.dumps(
            {
                "best_model_name": best_name,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_features": len(feature_cols),
                "feature_order": feature_cols,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 10) Guardar schema en el formato acordado (dict)
    _guardar_schema_dict(feature_cols)

    # 11) Salida por consola
    print("Train:", len(X_train), "| Test:", len(X_test), "| Features:", len(feature_cols))
    print("Métricas (test) por modelo:")
    for k in metrics:
        print(f"  {k:<16} -> {metrics[k]}")
    print("Mejor por RMSE:", best_name)
    print("Artefactos guardados en 'models/'.")


if __name__ == "__main__":
    main()
