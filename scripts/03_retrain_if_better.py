#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

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

def _hash_from_provenance():
    if PROVENANCE.exists():
        try:
            return json.loads(PROVENANCE.read_text(encoding="utf-8")).get("sha256")
        except Exception:
            return None
    return None

def _hash_from_metadata():
    if META.exists():
        try:
            return json.loads(META.read_text(encoding="utf-8")).get("data_sha256")
        except Exception:
            return None
    return None

def _current_best_rmse():
    if METRICS.exists():
        try:
            m = json.loads(METRICS.read_text(encoding="utf-8"))
            if "best" in m and "rmse" in m["best"]:
                return float(m["best"]["rmse"])
        except Exception:
            pass
    if META.exists():
        try:
            meta = json.loads(META.read_text(encoding="utf-8"))
            if "best_rmse" in meta:
                return float(meta["best_rmse"])
        except Exception:
            pass
    return float("inf")

def _preprocessor():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ],
        memory=None  # explicit to satisfy SonarCloud
    )

def _rmse(y_true, y_pred):
    # compatibilidad con versiones antiguas de sklearn
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred))
    }

def _train_models(X_train, y_train, X_test, y_test):
    scores = {}
    models = {}

    lr = LinearRegression().fit(X_train, y_train)
    scores["linear_regression"] = _evaluate(y_test, lr.predict(X_test))
    models["linear_regression"] = lr

    ridge = GridSearchCV(
        Ridge(),  # sin random_state
        {"alpha": [0.1, 1.0, 10.0, 100.0]},
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
    return scores, best_name, models[best_name]

from datetime import datetime, timezone

def main(force=False, test_size=0.2, random_state=42):
    if not PROCESSED.exists():
        raise FileNotFoundError(f"Falta {PROCESSED}. Ejecuta 01_validar_datos.py primero.")

    current_hash = _hash_from_provenance()
    have_prev = BEST_MODEL.exists() and PREPROCESSOR.exists() and META.exists()

    if have_prev:
        prev_hash = _hash_from_metadata()
        if (prev_hash == current_hash) and (not force):
            print("Artefactos previos encontrados y el hash de datos no cambió. No se reentrena.")
            return
        else:
            print("Se reentrenará (hash cambió o --force).")
    else:
        print("No hay artefactos previos completos. Entrenamiento inicial.")

    df = pd.read_csv(PROCESSED)
    target = "MEDV"
    features = [c for c in df.columns if c != target]

    pre = _preprocessor()
    x_proc = pre.fit_transform(df[features])
    X_train, X_test, y_train, y_test = train_test_split(
        x_proc, df[target], test_size=test_size, random_state=random_state
    )

    scores, best_name, best_model = _train_models(X_train, y_train, X_test, y_test)
    new_best_rmse = scores[best_name]["rmse"]
    prev_best_rmse = _current_best_rmse() if have_prev else float("inf")

    print(f"RMSE nuevo: {new_best_rmse:.4f} | RMSE actual: {prev_best_rmse:.4f}")

    timestamp = datetime.now(timezone.utc).isoformat()

    if (not have_prev) or (new_best_rmse < prev_best_rmse):
        dump(best_model, BEST_MODEL)
        dump(pre, PREPROCESSOR)

        METRICS.write_text(
            json.dumps({"per_model": scores,
                        "best": {"name": best_name, "rmse": new_best_rmse},
                        "evaluated_at": timestamp}, indent=2),
            encoding="utf-8"
        )

        META.write_text(
            json.dumps({
                "best_model_name": best_name,
                "best_rmse": new_best_rmse,
                "feature_order": features,
                "target": target,
                "trained_at": timestamp,
                "data_sha256": current_hash,
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test))
            }, indent=2),
            encoding="utf-8"
        )
        print("Modelo guardado como mejor actual en models/.")
    else:
        cand_dir = MODELS / "candidates"
        cand_dir.mkdir(exist_ok=True)
        prefix = cand_dir / f"candidate_{timestamp.replace(':','-')}"
        dump(best_model, str(prefix) + "_model.pkl")
        dump(pre, str(prefix) + "_preprocessor.pkl")
        Path(str(prefix) + "_metrics.json").write_text(
            json.dumps({
                "per_model": scores,
                "best": {"name": best_name, "rmse": new_best_rmse},
                "evaluated_at": timestamp,
                "note": "No reemplaza al actual (no mejora RMSE)."
            }, indent=2),
            encoding="utf-8"
        )
        print("El nuevo modelo no superó al actual. Guardado como candidato en models/candidates/.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Forzar reentrenamiento.")
    args = ap.parse_args()
    main(force=args.force)
