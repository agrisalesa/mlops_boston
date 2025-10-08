#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

PROCESSED = DATA / "processed" / "housing_clean.csv"
PROVENANCE = DATA / "provenance.json"

BEST_MODEL = MODELS / "best_model.pkl"
PREPROCESSOR = MODELS / "preprocessor.pkl"
METRICS = MODELS / "metrics.json"
META = MODELS / "model_metadata.json"
SCHEMA = MODELS / "schema.json"

TARGET = "MEDV"

def _rmse(y_true, y_pred):
    try:
        from sklearn.metrics import mean_squared_error as mse
        return float(mse(y_true, y_pred, squared=False))
    except TypeError:
        from sklearn.metrics import mean_squared_error as mse
        return float(np.sqrt(mse(y_true, y_pred)))

def _evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred))
    }

def _preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

def main(test_size=0.2, random_state=42):
    if not PROCESSED.exists():
        raise FileNotFoundError(f"Falta {PROCESSED}. Ejecuta 01_validar_datos.py primero.")

    df = pd.read_csv(PROCESSED)
    features = [c for c in df.columns if c != TARGET]

    pre = _preprocessor()
    X_proc = pre.fit_transform(df[features])
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, df[TARGET], test_size=test_size, random_state=random_state
    )

    scores = {}
    models = {}

    lr = LinearRegression().fit(X_train, y_train)
    scores["linear_regression"] = _evaluate(y_test, lr.predict(X_test))
    models["linear_regression"] = lr

    ridge = GridSearchCV(
        Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]},
        cv=5, n_jobs=-1
    ).fit(X_train, y_train)
    scores["ridge"] = _evaluate(y_test, ridge.predict(X_test))
    models["ridge"] = ridge.best_estimator_

    rf = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [200, 400],
         "max_depth": [None, 10, 20],
         "min_samples_leaf": [1, 2]},
        cv=3, n_jobs=-1
    ).fit(X_train, y_train)
    scores["random_forest"] = _evaluate(y_test, rf.predict(X_test))
    models["random_forest"] = rf.best_estimator_

    if HAS_XGB:
        xgb = GridSearchCV(
            XGBRegressor(
                random_state=42, objective="reg:squarederror",
                tree_method="hist", n_jobs=-1
            ),
            {"n_estimators": [300, 500],
             "max_depth": [4, 6, 8],
             "learning_rate": [0.05, 0.1],
             "subsample": [0.8, 1.0]},
            cv=3, n_jobs=-1
        ).fit(X_train, y_train)
        scores["xgboost"] = _evaluate(y_test, xgb.predict(X_test))
        models["xgboost"] = xgb.best_estimator_
    else:
        print("Aviso: XGBoost no está instalado; se omite.")

    best_name = min(scores.items(), key=lambda kv: kv[1]["rmse"])[0]
    best_model = models[best_name]

    # Guardar artefactos principales
    MODELS.mkdir(exist_ok=True)
    dump(best_model, BEST_MODEL)
    dump(pre, PREPROCESSOR)
    print("Guardado: best_model.pkl y preprocessor.pkl")

    # Guardar esquema simple (orden/ tipos básicos)
    schema = {
        "target": TARGET,
        "features": features,
        "dtypes": {c: ("float" if np.issubdtype(df[c].dtype, np.number) else "object") for c in df.columns if c != TARGET}
    }
    SCHEMA.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Cargar hash desde provenance.json (generado por 00_provenance.py)
    data_hash = None
    if PROVENANCE.exists():
        try:
            data_hash = json.loads(PROVENANCE.read_text(encoding="utf-8")).get("sha256")
        except Exception:
            pass

    # Guardar métricas y metadatos (clave para el retrain posterior)
    ts = datetime.now(timezone.utc).isoformat()
    METRICS.write_text(
        json.dumps({
            "per_model": scores,
            "best": {"name": best_name, "rmse": scores[best_name]["rmse"]},
            "evaluated_at": ts
        }, indent=2),
        encoding="utf-8"
    )

    META.write_text(
        json.dumps({
            "best_model_name": best_name,
            "best_rmse": scores[best_name]["rmse"],
            "feature_order": features,
            "target": TARGET,
            "trained_at": ts,
            "data_sha256": data_hash,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test))
        }, indent=2),
        encoding="utf-8"
    )

    # Resumen por consola
    print("\nEntrenamiento con grid completado.")
    print(f"Train: {len(y_train)} | Test: {len(y_test)} | Features: {len(features)}")
    print("Métricas (test) por modelo:")
    for k, v in scores.items():
        print(f"  {k:<16} -> {v}")
    print(f"Mejor por RMSE: {best_name}")
    print("Guardados: metrics.json, model_metadata.json y schema.json")

if __name__ == "__main__":
    main()
