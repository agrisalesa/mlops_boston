# tests/conftest.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # <- añade la raíz del repo al path

import json
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

PROCESSED = DATA / "processed" / "housing_clean.csv"
SCHEMA = MODELS / "schema.json"

@pytest.fixture(scope="session")
def processed_df():
    assert PROCESSED.exists(), "Falta data/processed/housing_clean.csv. Ejecuta 01_validar_datos.py en CI antes de las pruebas."
    df = pd.read_csv(PROCESSED)
    assert len(df) > 0, "El processed CSV no debe estar vacío."
    return df

@pytest.fixture(scope="session")
def schema():
    assert SCHEMA.exists(), "Falta models/schema.json. Debe generarse en 02_train_baseline.py."
    return json.loads(SCHEMA.read_text(encoding="utf-8"))

@pytest.fixture(scope="session")
def sample_payload(processed_df, schema):
    target = schema.get("target", "MEDV")
    feats = schema.get("features", [c for c in processed_df.columns if c != target])
    med = processed_df[feats].median(numeric_only=True)
    payload = {c: float(med[c]) if c in med else 0.0 for c in feats}
    for k, v in payload.items():
        if k in ("RAD", "TAX", "CHAS"):
            payload[k] = int(round(v))
        if k == "CHAS":
            payload[k] = 1 if payload[k] >= 1 else 0
    return payload

