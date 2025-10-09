#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detección de data drift con salvaguardas para muestras pequeñas.

Compara:
  - Referencia: data/processed/housing_clean.csv (features)
  - Vivo: logs/predictions.csv (últimas N filas)

Requisitos mínimos:
  - Al menos --min-live filas en el log vivo (default: 200)
  - PSI usa bins por cuantiles del set de referencia; si hay <5 cortes únicos, se omite columna
  - Se ignoran columnas no numéricas y metadatos (__pred, __ts)

Salida:
  - logs/drift_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = ROOT / "logs"
DRIFT_SUMMARY_HEADER = "=== Drift summary ==="
PROCESSED = DATA / "processed" / "housing_clean.csv"
PRED_LOG = LOGS / "predictions.csv"
REPORT = LOGS / "drift_report.json"

TARGET = "MEDV"
META_COLS = {"__pred", "__ts"}

def psi(expected, actual, bins=10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return np.nan

    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, qs))
    # si no hay al menos 5 cortes únicos, no tiene sentido el PSI
    if cuts.size < 5:
        return np.nan

    e = np.histogram(expected, bins=cuts)[0].astype(float)
    a = np.histogram(actual, bins=cuts)[0].astype(float)

    esum, asum = e.sum(), a.sum()
    if esum == 0 or asum == 0:
        return np.nan

    e = e / esum
    a = a / asum
    e = np.where(e == 0, 1e-6, e)
    a = np.where(a == 0, 1e-6, a)

    return float(np.sum((a - e) * np.log(a / e)))

def ks(expected, actual):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return np.nan, np.nan
    stat, p = ks_2samp(expected, actual)
    return float(stat), float(p)

def main(window=1000, min_live=200, psi_severe=0.2, ks_p_th=0.01):
    if not PROCESSED.exists():
        raise FileNotFoundError(f"Falta {PROCESSED}. Ejecuta 01_validar_datos.py primero.")

    ref = pd.read_csv(PROCESSED)
    ref = ref.drop(columns=[TARGET], errors="ignore")

    # No hay logs en vivo => no se puede evaluar drift
    if not PRED_LOG.exists():
        report = {
            "n_ref": int(len(ref)),
            "n_live": 0,
            "columns": {},
            "summary": {"drift_columns": [], "drift_detected": False, "reason": "no_live_data"}
        }
        REPORT.parent.mkdir(exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(DRIFT_SUMMARY_HEADER)
        print(json.dumps(report["summary"], indent=2))
        print(f"Reporte: {REPORT}")
        return

    live = pd.read_csv(PRED_LOG).tail(window)

    # Filtrar intersección de columnas y quitar metadatos
    common = [c for c in ref.columns if c in live.columns and c not in META_COLS]
    ref = ref[common]
    live = live[common]

    # Si no hay suficientes filas vivas, no evaluamos
    if len(live) < min_live:
        report = {
            "n_ref": int(len(ref)),
            "n_live": int(len(live)),
            "columns": {},
            "summary": {
                "drift_columns": [],
                "drift_detected": False,
                "reason": f"insufficient_live_rows(<{min_live})"
            }
        }
        REPORT.parent.mkdir(exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(DRIFT_SUMMARY_HEADER)
        print(json.dumps(report["summary"], indent=2))
        print(f"Reporte: {REPORT}")
        return

    drift_cols = []
    per_col = {}

    for col in common:
        if not np.issubdtype(ref[col].dtype, np.number):
            continue

        p = psi(ref[col].values, live[col].values, bins=10)
        kstat, kp = ks(ref[col].values, live[col].values)

        per_col[col] = {"psi": p, "ks_stat": kstat, "ks_pvalue": kp}

        severe = (not np.isnan(p) and p > psi_severe) or (not np.isnan(kp) and kp < ks_p_th)
        if severe:
            drift_cols.append(col)

    report = {
        "n_ref": int(len(ref)),
        "n_live": int(len(live)),
        "columns": per_col,
        "summary": {"drift_columns": drift_cols, "drift_detected": len(drift_cols) > 0}
    }

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(DRIFT_SUMMARY_HEADER)
    print(json.dumps(report["summary"], indent=2))
    print(f"Reporte: {REPORT}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=1000, help="Últimas N filas del log a evaluar")
    ap.add_argument("--min-live", type=int, default=200, help="Mínimo de filas live para evaluar drift")
    ap.add_argument("--psi-severe", type=float, default=0.2, help="Umbral PSI para drift severo")
    ap.add_argument("--ks-p-th", type=float, default=0.01, help="Umbral p-value KS (menor => drift)")
    args = ap.parse_args()
    main(window=args.window, min_live=args.min_live, psi_severe=args.psi_severe, ks_p_th=args.ks_p_th)
