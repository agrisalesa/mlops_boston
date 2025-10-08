# app/server.py
import json
import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Rutas base
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
SCHEMA_PATH = MODELS_DIR / "schema.json"  # generado en entrenamiento
PRED_LOG_PATH = LOGS_DIR / "predictions.csv"

# App Flask
app = Flask(__name__)

# Artefactos en memoria
_model = None
_preprocessor = None
_schema = None


def _load_artifacts():
    """
    Carga en memoria el modelo, el preprocesador y el esquema.
    Se ejecuta en el primer request o en /health.
    """
    global _model, _preprocessor, _schema

    # Cargar modelo
    if _model is None:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"No se encontró el modelo en {BEST_MODEL_PATH}")
        _model = joblib.load(BEST_MODEL_PATH)

    # Cargar preprocesador
    if _preprocessor is None:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"No se encontró el preprocesador en {PREPROCESSOR_PATH}")
        _preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Cargar esquema (opcional pero recomendado)
    if _schema is None:
        if SCHEMA_PATH.exists():
            _schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        else:
            # Si no hay schema.json, se sigue sin él (validación mínima)
            _schema = {}


def _coerce_and_align(payload: dict) -> pd.DataFrame:
    """
    Convierte el payload a DataFrame, alinea columnas contra el esquema si existe,
    fuerza tipos básicos y deja NaN en faltantes (el preprocesador se encarga).
    Ignora llaves extra no esperadas.
    """
    if not isinstance(payload, dict):
        raise ValueError("El cuerpo debe ser un objeto JSON con pares clave-valor de features.")

    df = pd.DataFrame([payload])

    # Si hay esquema, reordenar y añadir faltantes como NaN
    feature_order = None
    if isinstance(_schema, dict):
        feature_order = _schema.get("features") or _schema.get("feature_order")
    if feature_order:
        # Mantener solo columnas conocidas y en orden; agregar faltantes como NaN
        cols_known = [c for c in feature_order if c in df.columns]
        cols_missing = [c for c in feature_order if c not in df.columns]
        df = df[cols_known].copy()
        for c in cols_missing:
            df[c] = np.nan
        # reordenar exactamente como el entrenamiento
        df = df[feature_order]

    # Forzar tipos básicos cuando sea posible (sin romper; NaN cuando falle)
    # El preprocesador (imputer + scaler) se encargará de los NaN remanentes.
    if isinstance(_schema, dict) and "dtypes" in _schema:
        dtypes = _schema["dtypes"]
        for col, kind in dtypes.items():
            if col not in df.columns:
                continue
            # Intento de conversión por categoría
            try:
                if kind in ("float", "float64", "number"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif kind in ("int", "int64"):
                    # primero a float para permitir NaN, luego a Int64 si se requiere
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif kind in ("bool", "binary"):
                    # normalizar a 0/1 si vienen como texto
                    df[col] = df[col].map({True: 1, False: 0, "1": 1, "0": 0}).astype("float64")
                else:
                    # texto u otros
                    df[col] = df[col].astype("object")
            except Exception:
                # si falla, dejar como está; el preprocesador resolverá con NaN donde aplique
                pass

    return df


def _log_prediction(df_features: pd.DataFrame, pred: float):
    """
    Registra en logs/predictions.csv las features recibidas con la predicción y timestamp.
    Este archivo se usa para medir drift en producción.
    """
    LOGS_DIR.mkdir(exist_ok=True)
    row = df_features.copy()
    row["__pred"] = float(pred)
    row["__ts"] = datetime.utcnow().isoformat() + "Z"

    # Escribir con encabezado si el archivo no existe
    header = not PRED_LOG_PATH.exists()
    row.to_csv(PRED_LOG_PATH, mode="a", header=header, index=False)


@app.route("/health", methods=["GET"])
def health():
    """
    Verifica disponibilidad de artefactos y que se pueden cargar en memoria.
    """
    try:
        _load_artifacts()
        return jsonify({
            "status": "ok",
            "best_model": BEST_MODEL_PATH.exists(),
            "preprocessor": PREPROCESSOR_PATH.exists(),
            "schema": SCHEMA_PATH.exists(),
        })
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Recibe un JSON con las features del modelo y responde con la predicción.
    Ejemplo de input:
    {
      "CRIM": 0.1, "ZN": 18, "INDUS": 2.31, "CHAS": 0, "NOX": 0.538, "RM": 6.575,
      "AGE": 65.2, "DIS": 4.09, "RAD": 1, "TAX": 296, "PTRATIO": 15.3, "B": 396.9, "LSTAT": 4.98
    }
    """
    try:
        _load_artifacts()

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "El cuerpo debe ser JSON válido."}), 400

        # Permitir uno o varios registros; normalizar a lista de dicts
        if isinstance(payload, list):
            if len(payload) == 0:
                return jsonify({"error": "La lista de registros está vacía."}), 400
            # Para este servidor, procesamos de a uno; podría ampliarse a batch
            payload = payload[0]

        df = _coerce_and_align(payload)

        # Transformación y predicción
        X = _preprocessor.transform(df)
        yhat = _model.predict(X)
        pred = float(np.ravel(yhat)[0])

        # Logging para drift
        try:
            _log_prediction(df, pred)
        except Exception as log_err:
            app.logger.warning(f"No se pudo registrar la predicción: {log_err}")

        return jsonify({"prediction": pred})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        # Mensaje simple pero suficiente para debug
        return jsonify({"error": f"Fallo en la inferencia: {str(e)}"}), 500


if __name__ == "__main__":
    # Para ejecución directa sin gunicorn (desarrollo)
    # En producción el arranque lo hace gunicorn desde el Dockerfile:
    # CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app.server:app"]
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
