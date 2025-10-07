# scripts/02_train_baseline.py
# Entrenamiento con varios modelos y grid search.
# Modelos: LinearRegression, Ridge (grid), RandomForest (grid), XGBoost (grid).
# Guarda solo el mejor modelo y el preprocesador; reporta métricas de todos.

import os
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


def leer_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_dataset(path_csv):
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"No se encontró el dataset limpio en: {path_csv}")
    return pd.read_csv(path_csv)


def cargar_esquema(path_schema):
    if not os.path.exists(path_schema):
        raise FileNotFoundError(f"Falta el esquema de columnas: {path_schema}")
    return leer_json(path_schema)


def cargar_provenance(path_prov):
    if os.path.exists(path_prov):
        return leer_json(path_prov)
    return {}


def separar_xy(df, esquema):
    target = esquema["target"]
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no está en el dataset.")
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    return X, y


def preprocesador_minimo():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def evaluar(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred))
    }


def guardar_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    # Rutas
    path_clean = os.path.join("data", "processed", "housing_clean.csv")
    path_schema = os.path.join("models", "schema.json")
    path_prov = os.path.join("data", "provenance.json")
    out_dir = "models"

    # Parámetros
    seed = 42
    test_size = 0.2
    cv_folds = 5
    scoring = "neg_mean_squared_error"

    # Insumos
    df = cargar_dataset(path_clean)
    esquema = cargar_esquema(path_schema)
    provenance = cargar_provenance(path_prov)

    # Train/Test
    X, y = separar_xy(df, esquema)
    feature_order = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Preprocesamiento
    prep = preprocesador_minimo()
    X_train_p = prep.fit_transform(X_train)
    X_test_p = prep.transform(X_test)

    # ==== Modelos y grids ====
    modelos = {}

    # 1) LinearRegression
    mdl_lin = LinearRegression()
    mdl_lin.fit(X_train_p, y_train)
    modelos["linear_regression"] = mdl_lin

    # 2) Ridge con grid de alphas habituales
    ridge = Ridge(random_state=seed)
    ridge_grid = {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    }
    ridge_gs = GridSearchCV(ridge, ridge_grid, cv=cv_folds, scoring=scoring, n_jobs=-1, verbose=0)
    ridge_gs.fit(X_train_p, y_train)
    modelos["ridge"] = ridge_gs.best_estimator_

    # 3) RandomForest con grid
    rf = RandomForestRegressor(random_state=seed, n_jobs=-1)
    rf_grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }
    rf_gs = GridSearchCV(rf, rf_grid, cv=cv_folds, scoring=scoring, n_jobs=-1, verbose=0)
    rf_gs.fit(X_train_p, y_train)
    modelos["random_forest"] = rf_gs.best_estimator_

    # 4) XGBoost con grid
    xgb = XGBRegressor(
        random_state=seed,
        tree_method="hist",
        n_jobs=-1,
        objective="reg:squarederror"
    )
    xgb_grid = {
        "n_estimators": [300, 600],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_lambda": [1.0, 2.0]
    }
    xgb_gs = GridSearchCV(xgb, xgb_grid, cv=cv_folds, scoring=scoring, n_jobs=-1, verbose=0)
    xgb_gs.fit(X_train_p, y_train)
    modelos["xgboost"] = xgb_gs.best_estimator_

    # ==== Evaluación en test ====
    metrics = {}
    for name, model in modelos.items():
        yhat = model.predict(X_test_p)
        metrics[name] = evaluar(y_test, yhat)

    # Selección por RMSE (menor mejor)
    def rmse_of(m): return metrics[m]["rmse"]
    best_name = min(metrics.keys(), key=rmse_of)
    best_model = modelos[best_name]
    best_metrics = metrics[best_name]

    # ==== Persistencia (solo el mejor) ====
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(prep, os.path.join(out_dir, "preprocessor.pkl"))
    joblib.dump(best_model, os.path.join(out_dir, "best_model.pkl"))

    guardar_json({
        "metrics": metrics,
        "selection": {"criterion": "rmse", "best": best_name}
    }, os.path.join(out_dir, "metrics.json"))

    guardar_json({
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "test_size": test_size,
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "feature_order": feature_order,
        "target": esquema["target"],
        "data_provenance": {
            "sha256": provenance.get("sha256"),
            "source": provenance.get("source"),
            "generated_at": provenance.get("generated_at")
        },
        "best_model": {
            "name": best_name,
            "path": "models/best_model.pkl",
            "test_metrics": best_metrics
        }
    }, os.path.join(out_dir, "model_metadata.json"))

    print("\nEntrenamiento con grid completado.")
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X.shape[1]}")
    print("Métricas (test) por modelo:")
    for name in sorted(metrics.keys()):
        print(f"  {name:16s} -> {metrics[name]}")
    print("Mejor por RMSE:", best_name)
    print("Guardado: best_model.pkl y preprocessor.pkl")


if __name__ == "__main__":
    main()
