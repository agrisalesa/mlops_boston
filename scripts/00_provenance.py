# scripts/00_provenance.py
# Objetivo:
#   - Calcular el hash SHA256 del dataset crudo (data/raw/housing.csv)
#   - Generar DATA_PROVENANCE.md (humano) y data/provenance.json (máquina)
#   - No altera los datos; solo registra trazabilidad

import hashlib
import json
import os
from datetime import datetime

def calcular_hash(ruta_archivo):
    """Calcula el hash SHA256 de un archivo de forma incremental para no cargarlo completo en memoria."""
    sha = hashlib.sha256()
    with open(ruta_archivo, "rb") as f:
        for bloque in iter(lambda: f.read(8192), b""):
            sha.update(bloque)
    return sha.hexdigest()

def guardar_md(ruta_dataset, fuente, hash_val, fecha, notas):
    """Escribe DATA_PROVENANCE.md orientado a humanos."""
    contenido = f"""# Procedencia del Dataset

- **Fuente:** {fuente}
- **Archivo:** {ruta_dataset}
- **Hash (SHA256):** {hash_val}
- **Generado en:** {fecha}
- **Notas:** {notas}

"""
    with open("DATA_PROVENANCE.md", "w", encoding="utf-8") as f:
        f.write(contenido)

def guardar_json(ruta_dataset, fuente, hash_val, fecha, notas):
    """Escribe data/provenance.json para consumo por scripts."""
    os.makedirs("data", exist_ok=True)
    payload = {
        "source": fuente,
        "path": ruta_dataset,
        "sha256": hash_val,
        "generated_at": fecha,
        "notes": notas
    }
    with open(os.path.join("data", "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def main():
    ruta_dataset = os.path.join("data", "raw", "housing.csv")
    if not os.path.exists(ruta_dataset):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_dataset}")

    fuente = "Kaggle – altavish/boston-housing-dataset"
    notas = "Uso educativo. Dataset clásico de regresión Boston Housing."
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hash_val = calcular_hash(ruta_dataset)
    guardar_md(ruta_dataset, fuente, hash_val, fecha, notas)
    guardar_json(ruta_dataset, fuente, hash_val, fecha, notas)

    print("Trazabilidad actualizada.")
    print("SHA256:", hash_val)
    print("Archivos generados: DATA_PROVENANCE.md y data/provenance.json")

if __name__ == "__main__":
    main()
