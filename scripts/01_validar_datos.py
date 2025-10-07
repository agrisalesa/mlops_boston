# scripts/01_validar_datos.py
# Validación e imputación genérica:
# - Mediana para todas las columnas float
# - Moda para columnas binarias (0/1)
# - Mediana + cast para columnas int no binarias
# - Fuerza tipos según schema.json y valida hash de provenance si existe

import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime

RAW_PATH = os.path.join("data", "raw", "housing.csv")
PROC_PATH = os.path.join("data", "processed", "housing_clean.csv")
SCHEMA_PATH = os.path.join("models", "schema.json")
PROV_JSON = os.path.join("data", "provenance.json")

# ---------------- utilidades ----------------

def leer_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def validar_hash(path_csv, path_prov):
    if not os.path.exists(path_prov):
        return  # no hay control de hash, continuar
    prov = leer_json(path_prov)
    esperado = prov.get("sha256")
    if not esperado:
        return
    actual = sha256_file(path_csv)
    if actual != esperado:
        raise ValueError(
            "El hash del dataset no coincide con provenance.json.\n"
            f"  esperado: {esperado}\n  actual:   {actual}\n"
            "Verifica que no cambió el CSV o vuelve a generar provenance."
        )

def coaccionar_a_numerico(df, schema):
    # Convierte todas las columnas del esquema a numéricas con coerción
    # (deja NaN si no se puede convertir, que luego se imputan).
    for col, spec in schema["features"].items():
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")
        tipo = spec["type"]
        if tipo == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif tipo == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            raise ValueError(f"Tipo no soportado en schema para {col}: {tipo}")
    # target también debe ser numérico
    tgt = schema["target"]
    if tgt not in df.columns:
        raise ValueError(f"Falta la columna objetivo '{tgt}'.")
    df[tgt] = pd.to_numeric(df[tgt], errors="coerce")
    return df

def es_binaria(serie):
    # Detecta binaria si los valores únicos no NaN ⊆ {0,1}
    vals = pd.unique(serie.dropna())
    if len(vals) == 0:
        return False
    return set(np.unique(vals)).issubset({0, 1})

def imputar_generico(df, schema, reporte):
    """
    Reglas:
      - float: mediana
      - int binaria (0/1): moda
      - int no binaria: mediana y cast a int al final
      - target: si tiene NaN, se eliminan esas filas (no se imputa la Y)
    """
    target = schema["target"]

    # 1) Imputación por tipo para features
    for col, spec in schema["features"].items():
        if col == target:
            continue
        tipo = spec["type"]
        n_nulls = int(df[col].isna().sum())
        if n_nulls == 0:
            continue

        if tipo == "float":
            med = float(df[col].median(skipna=True))
            df[col] = df[col].fillna(med)
            reporte.append(f"- {col}: imputados {n_nulls} NaN con MEDIANA (float) = {med}")
        elif tipo == "int":
            if es_binaria(df[col]):
                # moda (en binaria, prioriza 0 en empate por estabilidad)
                moda = int(df[col].mode(dropna=True).iloc[0]) if not df[col].mode(dropna=True).empty else 0
                df[col] = df[col].fillna(moda)
                reporte.append(f"- {col}: imputados {n_nulls} NaN con MODA (binaria) = {moda}")
            else:
                med = float(df[col].median(skipna=True))
                df[col] = df[col].fillna(med)
                reporte.append(f"- {col}: imputados {n_nulls} NaN con MEDIANA (int) = {med}")
        else:
            # por si el schema trae algo raro
            raise ValueError(f"Tipo no soportado para imputación en {col}: {tipo}")

    # 2) Cast final de enteros (después de imputar)
    for col, spec in schema["features"].items():
        if spec["type"] == "int":
            # redondeo defensivo antes de cast
            df[col] = np.round(df[col]).astype("Int64")  # Int64 tolera NaN temporales
            df[col] = df[col].astype(int)

    # 3) Filas con NaN en target -> eliminar
    y_nulls = int(df[target].isna().sum())
    if y_nulls > 0:
        df = df[~df[target].isna()].copy()
        reporte.append(f"- {target}: filas con NaN en la variable objetivo eliminadas = {y_nulls}")

    return df

def imprimir_resumen(df, reporte):
    print("\n==== Reporte de imputación ====")
    if reporte:
        for r in reporte:
            print(r)
    else:
        print("No hubo imputaciones necesarias.")

    print("\n==== Forma del DataFrame ====")
    print(df.shape)

    print("\n==== Tipos de datos ====")
    print(df.dtypes)

    print("\n==== Nulos por columna ====")
    print(df.isna().sum())

    print("\n==== Estadísticos descriptivos ====")
    print(df.describe().T)

# ---------------- flujo principal ----------------

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de datos crudos: {RAW_PATH}")
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Falta el schema: {SCHEMA_PATH}")

    validar_hash(RAW_PATH, PROV_JSON)

    df = pd.read_csv(RAW_PATH)
    schema = leer_json(SCHEMA_PATH)

    # Asegurar nombres esperados (si tu pipeline renombra, hazlo aquí)
    # df.rename(columns={...}, inplace=True)

    # Coaccionar a numérico según schema (deja NaN donde no se pueda convertir)
    df = coaccionar_a_numerico(df, schema)

    # Imputación genérica por tipo (float=mediana, binaria=moda, int no binaria=mediana+cast)
    reporte = []
    df = imputar_generico(df, schema, reporte)

    # Guardado
    os.makedirs(os.path.dirname(PROC_PATH), exist_ok=True)
    df.to_csv(PROC_PATH, index=False)

    imprimir_resumen(df, reporte)
    print(f"\nArchivo limpio guardado en: {PROC_PATH}")
    print(f"Generado: {datetime.now().isoformat(timespec='seconds')}")

if __name__ == "__main__":
    main()
