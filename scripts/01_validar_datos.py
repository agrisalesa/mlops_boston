#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Rutas base del proyecto
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
SCHEMA_PATH = MODELS / "schema.json"


def _load_schema() -> Tuple[Dict[str, dict], str]:
    """
    Carga models/schema.json y devuelve:
      - features: dict {col: {"dtype": "...", "binary": bool}}
      - target: nombre de la columna objetivo

    Soporta dos formatos:
      A) {"target":"MEDV","features":{"CRIM":{"dtype":"float"},...}}
      B) {"target":"MEDV","features":["CRIM",...],"dtypes":{"CRIM":"float",...}}
    """
    raw = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    target = raw.get("target", "MEDV")

    feats = raw.get("features", {})
    dtypes_map = raw.get("dtypes", {})

    if isinstance(feats, list):
        # Formato B => convertir a dict usando dtypes
        features = {}
        for c in feats:
            features[c] = {
                "dtype": str(dtypes_map.get(c, "float")).lower(),
                "binary": False if c != "CHAS" else True,
            }
    elif isinstance(feats, dict):
        # Formato A => asegurar defaults
        features = {}
        for c, spec in feats.items():
            features[c] = {
                "dtype": str(spec.get("dtype", "float")).lower(),
                "binary": bool(spec.get("binary", False)),
            }
    else:
        raise ValueError("Formato de schema['features'] no soportado.")

    return features, target


def _coaccionar_a_numerico(df: pd.DataFrame,
                           schema_features: Dict[str, dict],
                           target: str) -> pd.DataFrame:
    """
    Fuerza tipos numéricos según el esquema.
    - Mantiene features y también el target.
    - Agrega faltantes como NaN (en features).
    - 'binary' (ej. CHAS) se mapea a 0/1.
    - Reordena: primero features (según esquema) y al final el target.
    """
    feature_order = list(schema_features.keys())

    # Columnas que queremos conservar: features + target
    cols_keep = feature_order + [target]

    # Mantener solo esperadas + target si existen en df
    df = df[[c for c in df.columns if c in cols_keep]].copy()

    # Agregar features faltantes como NaN
    for c in feature_order:
        if c not in df.columns:
            df[c] = np.nan

    # Si el target no está, lo dejamos tal cual (lo validamos luego en main)
    # Forzar tipos de features
    for col, spec in schema_features.items():
        kind = str(spec.get("dtype", "float")).lower()
        try:
            if spec.get("binary", False) or kind in ("bool", "binary"):
                df[col] = df[col].map({True: 1, False: 0, "1": 1, "0": 0})
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif kind in ("int", "int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float")
            elif kind in ("float", "float64", "number"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reorden: primero features y al final el target (si existe)
    ordered = feature_order + ([target] if target in df.columns else [])
    df = df[ordered]
    return df


def _imputar(df: pd.DataFrame, schema_features: Dict[str, dict]) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Imputa:
      - Columnas binarias: moda (0/1).
      - Resto numéricas: mediana.
    Devuelve df imputado + reporte de imputación.
    """
    reporte = {}
    for col, spec in schema_features.items():
        vals_antes = df[col].isna().sum()
        if vals_antes == 0:
            continue

        if spec.get("binary", False):
            # Moda binaria
            moda = df[col].mode(dropna=True)
            moda = int(moda.iloc[0]) if not moda.empty else 0
            df[col] = df[col].fillna(moda)
            reporte[col] = {
                "metodo": "MODA (binaria)",
                "valor": moda,
                "imputados": int(vals_antes),
            }
        else:
            # Mediana
            med = float(df[col].median(skipna=True))
            df[col] = df[col].fillna(med)
            reporte[col] = {
                "metodo": "MEDIANA (float)",
                "valor": med,
                "imputados": int(vals_antes),
            }

    return df, reporte


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1) Cargar schema y dataset crudo
    schema_features, target = _load_schema()
    raw_csv = RAW / "housing.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"No se encuentra {raw_csv}")

    df = pd.read_csv(raw_csv)

    # 2) Coerción a numérico según schema
    df = _coaccionar_a_numerico(df, schema_features, target)

    # 3) Validación básica de target
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no está presente en el dataset crudo.")

    # 4) Imputación
    df_imputed, reporte = _imputar(df, schema_features)

    # 5) Guardar
    out_path = PROCESSED / "housing_clean.csv"
    df_imputed.to_csv(out_path, index=False)

    # 6) Reporte
    print("\n==== Reporte de imputación ====")
    for k, v in reporte.items():
        print(f"- {k}: imputados {v['imputados']} con {v['metodo']} = {v['valor']}")

    print("\n==== Forma del DataFrame ====")
    print(df_imputed.shape)

    print("\n==== Tipos de datos ====")
    print(df_imputed.dtypes)

    print("\n==== Nulos por columna ====")
    print(df_imputed.isna().sum())

    print("\n==== Estadísticos descriptivos ====")
    desc = df_imputed.describe().T
    desc = desc.rename(
        columns={
            "count": "count", "mean": "mean", "std": "std",
            "min": "min", "25%": "25%", "50%": "50%", "75%": "75%", "max": "max"
        }
    )
    print(desc)

    print(f"\nArchivo limpio guardado en: {out_path}")
    print(f"Generado: {pd.Timestamp.utcnow().isoformat()}")


if __name__ == "__main__":
    main()
