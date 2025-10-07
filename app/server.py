import os
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# --- Rutas base del proyecto ---
APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(APP_ROOT, "models")
LOGS_DIR = os.path.join(APP_ROOT, "logs")

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")
BEST_MODEL_PATH   = os.path.join(MODELS_DIR, "best_model.pkl")
SCHEMA_PATH       = os.path.join(MODELS_DIR, "schema.json")
METADATA_PATH     = os.path.join(MODELS_DIR, "model_metadata.json")
PRED_LOG_PATH     = os.path.join(LOGS_DIR, "predictions.csv")

# Reglas de validación por columna (rangos razonables para Boston Housing)
RANGE_RULES = {
    "CRIM":   (0.0, 100.0),
    "ZN":     (0.0, 100.0),
    "INDUS":  (0.0, 30.0),
    "CHAS":   (0, 1),        # binaria
    "NOX":    (0.3, 1.0),
    "RM":     (3.0, 9.0),
    "AGE":    (0.0, 100.0),
    "DIS":    (0.5, 15.0),
    "RAD":    (1, 24),       # índice entero positivo
    "TAX":    (100.0, 800.0),
    "PTRATIO":(10.0, 30.0),
    "B":      (0.0, 400.0),
    "LSTAT":  (0.0, 40.0)
}

app = Flask(__name__)

# --- Utilidades de carga ---
def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    schema = _load_json(SCHEMA_PATH)
    feature_order = None
    if os.path.exists(METADATA_PATH):
        meta = _load_json(METADATA_PATH)
        feature_order = meta.get("feature_order")
    return preprocessor, model, schema, feature_order

# Artefactos en memoria al iniciar
PREPROCESSOR = None
MODEL = None
SCHEMA = None
FEATURE_ORDER = None
STARTUP_ERROR = None
try:
    PREPROCESSOR, MODEL, SCHEMA, FEATURE_ORDER = _safe_load_artifacts()
except Exception as e:
    STARTUP_ERROR = str(e)

# --- Validación y logging ---
def _validate_payload(payload, schema):
    required = list(schema["features"].keys())
    if not isinstance(payload, dict):
        return False, "El cuerpo debe ser un objeto JSON.", None

    missing = [c for c in required if c not in payload]
    if missing:
        return False, f"Faltan columnas: {missing}", None

    clean = {}
    errors = []

    for c in required:
        raw = payload[c]
        decl = schema["features"][c]["type"]

        # casteo defensivo según schema
        try:
            val = int(float(raw)) if decl == "int" else float(raw)
        except Exception:
            errors.append(f"{c}: no convertible a {decl}")
            continue

        # reglas de rango
        if c in RANGE_RULES:
            lo, hi = RANGE_RULES[c]
            if not (lo <= val <= hi):
                errors.append(f"{c}: fuera de rango [{lo}, {hi}] -> {val}")

        # reglas específicas
        if c == "CHAS" and val not in (0, 1):
            errors.append("CHAS: debe ser 0 o 1")
        if c == "RAD" and (int(val) != val or val < 1):
            errors.append("RAD: debe ser entero positivo (>=1)")

        clean[c] = val

    if errors:
        return False, "; ".join(errors), None

    return True, "", clean

def _log_prediction(input_df, pred_value):
    os.makedirs(os.path.dirname(PRED_LOG_PATH), exist_ok=True)
    row = input_df.copy()
    row["prediction"] = float(pred_value)
    row["ts"] = datetime.now(timezone.utc).isoformat()
    header = not os.path.exists(PRED_LOG_PATH)
    row.to_csv(PRED_LOG_PATH, mode="a", header=header, index=False)

# --- Endpoints ---
def health_endpoint():
    status = {
        "status": "ok" if STARTUP_ERROR is None else "error",
        "preprocessor": os.path.exists(PREPROCESSOR_PATH),
        "best_model": os.path.exists(BEST_MODEL_PATH),
        "schema": os.path.exists(SCHEMA_PATH),
    }
    code = 200 if status["status"] == "ok" else 500
    if STARTUP_ERROR:
        status["detail"] = STARTUP_ERROR
    return jsonify(status), code

def predict_endpoint():
    if any(x is None for x in [PREPROCESSOR, MODEL, SCHEMA]):
        return jsonify({"error": "Artefactos no disponibles. Revisa /health."}), 500

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSON inválido."}), 400

    ok, msg, clean = _validate_payload(payload, SCHEMA)
    if not ok:
        return jsonify({"error": msg}), 422  # Unprocessable Entity

    try:
        cols = FEATURE_ORDER if FEATURE_ORDER else list(SCHEMA["features"].keys())
        X = pd.DataFrame([[clean[c] for c in cols]], columns=cols)
        Xp = PREPROCESSOR.transform(X)
        yhat = MODEL.predict(Xp)[0]
    except Exception as e:
        return jsonify({"error": f"Error en predicción: {str(e)}"}), 500

    try:
        _log_prediction(X, yhat)
    except Exception:
        # el logging nunca debe tumbar la respuesta
        pass

    return jsonify({"prediction": float(yhat)}), 200

# Registro funcional de rutas
app.add_url_rule("/health",  endpoint="health",  view_func=health_endpoint,  methods=["GET"])
app.add_url_rule("/predict", endpoint="predict", view_func=predict_endpoint, methods=["POST"])

if __name__ == "__main__":
    # Para desarrollo local; cambiar host/port
    app.run(host="127.0.0.1", port=8000, debug=True)
