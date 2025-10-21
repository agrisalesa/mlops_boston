#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pickle
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# ✅ Ajuste universal de rutas (funciona dentro y fuera de Docker)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(BASE_DIR).resolve().parents[0]  # sube a la raíz del proyecto (/app en Docker)
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

BEST_MODEL_PATH = MODELS / "best_model.pkl"
PREPROC_PATH = MODELS / "preprocessor.pkl"
SCHEMA_PATH = MODELS / "schema.json"
PRED_LOG = LOGS / "predictions.csv"


app = Flask(__name__)

# Carga artefactos al iniciar
_model = None
_preproc = None
_schema: Dict = {}

LOGS.mkdir(parents=True, exist_ok=True)


def _normalize_schema(schema: Dict) -> Dict:
    """
    Acepta dos formatos de schema y devuelve uno normalizado:
      {"target":"MEDV",
       "features":{"COL":{"dtype":"float","binary":false}, ...},
       "dtypes":{"COL":"float",...}}
    """
    if not isinstance(schema, dict):
        raise ValueError("schema.json inválido")

    feats = schema.get("features", {})
    dtypes = schema.get("dtypes", {})

    if isinstance(feats, list):
        # lista + dtypes => pasar a dict
        feats_dict = {}
        for c in feats:
            feats_dict[c] = {"dtype": str(dtypes.get(c, "float")).lower(), "binary": bool(c == "CHAS")}
        schema["features"] = feats_dict

    if "dtypes" not in schema and isinstance(schema.get("features"), dict):
        schema["dtypes"] = {c: spec.get("dtype", "float") for c, spec in schema["features"].items()}

    if "target" not in schema:
        schema["target"] = "MEDV"

    return schema


def _cast_and_align(payload: Dict) -> pd.DataFrame:
    """
    Construye un DataFrame de una sola fila con el orden del schema
    y castea numéricos. Columnas faltantes quedan en NaN.
    """
    feats_dict = _schema["features"]  # dict {col:{dtype,binary}}
    order = list(feats_dict.keys())

    row = {}
    for c in order:
        val = payload.get(c, None)
        if val is None:
            row[c] = np.nan
            continue

        kind = str(feats_dict[c].get("dtype", "float")).lower()
        if feats_dict[c].get("binary", False) or kind in ("bool", "binary"):
            row[c] = 1.0 if str(val) in ("1", "true", "True") else 0.0
        else:
            try:
                row[c] = float(val)
            except Exception:
                row[c] = np.nan

    return pd.DataFrame([row], columns=order)


def _log_prediction(payload: Dict, pred: float):
    PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = payload.copy()
    rec["prediction"] = float(pred)
    rec_df = pd.DataFrame([rec])
    if PRED_LOG.exists():
        rec_df.to_csv(PRED_LOG, mode="a", index=False, header=False)
    else:
        rec_df.to_csv(PRED_LOG, index=False)


@app.route("/health", methods=["GET"])
def health():
    ok = BEST_MODEL_PATH.exists() and PREPROC_PATH.exists() and SCHEMA_PATH.exists()
    return jsonify(
        {
            "status": "ok" if ok else "missing_artifacts",
            "best_model": BEST_MODEL_PATH.exists(),
            "preprocessor": PREPROC_PATH.exists(),
            "schema": SCHEMA_PATH.exists(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Cuerpo JSON inválido"}), 400

    x_df = _cast_and_align(data)
    x_proc = _preproc.transform(x_df)
    yhat = _model.predict(x_proc)
    pred = float(yhat[0])

    _log_prediction(data, pred)
    return jsonify({"prediction": pred})


def _load_artifacts():
    global _model, _preproc, _schema
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {BEST_MODEL_PATH}")
    if not PREPROC_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {PREPROC_PATH}")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"No se encuentra {SCHEMA_PATH}")

    with open(BEST_MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    with open(PREPROC_PATH, "rb") as f:
        _preproc = pickle.load(f)

    raw = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    _schema = _normalize_schema(raw)


if __name__ == "__main__":
    _load_artifacts()
    app.run(host="0.0.0.0", port=8000, debug=True)
