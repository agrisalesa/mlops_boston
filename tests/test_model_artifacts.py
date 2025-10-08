# tests/test_model_artifacts.py
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

def test_artifacts_exist():
    assert (MODELS / "best_model.pkl").exists(), "Falta models/best_model.pkl"
    assert (MODELS / "preprocessor.pkl").exists(), "Falta models/preprocessor.pkl"
    assert (MODELS / "metrics.json").exists(), "Falta models/metrics.json"
    assert (MODELS / "model_metadata.json").exists(), "Falta models/model_metadata.json"
    assert (MODELS / "schema.json").exists(), "Falta models/schema.json"

def test_model_predicts(processed_df):
    # Carga artefactos
    model = joblib.load(MODELS / "best_model.pkl")
    pre = joblib.load(MODELS / "preprocessor.pkl")
    # Arma una fila de prueba con medianas
    X = processed_df.drop(columns=["MEDV"], errors="ignore")
    x_row = X.median(numeric_only=True).to_frame().T
    Xp = pre.transform(x_row)
    yhat = model.predict(Xp)
    assert np.isfinite(yhat).all(), "La predicción debe ser numérica y finita."
