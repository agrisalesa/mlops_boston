# Proyecto MLOps - Boston Housing

Autor: Andrés Grisales Ardila  
Repositorio: [https://github.com/agrisalesa/mlops_boston](https://github.com/agrisalesa/mlops_boston)

Implementación completa de un flujo MLOps de punta a punta para un modelo de regresión con el dataset Boston Housing, incluyendo:

- Control de versiones y trazabilidad de datos  
- Validación e imputación robusta  
- Entrenamiento automatizado y selección del mejor modelo  
- Monitoreo de data drift  
- Reentrenamiento condicional  
- API REST para predicciones  
- CI/CD completo con GitHub Actions y Docker  

---

## Estructura del proyecto

```
mlops_boston/
├── .github/workflows/        # Pipeline CI/CD (mlops.yml)
├── app/                      # API (server.py)
├── data/                     # Datos crudos y procesados
├── logs/                     # Logs y reportes de drift
├── models/                   # Modelos, preprocesador y metadatos
├── scripts/                  # Scripts principales del flujo
│   ├── 00_provenance.py
│   ├── 01_validar_datos.py
│   ├── 02_train_baseline.py
│   ├── 03_retrain_if_better.py
│   └── 04_drift_check.py
├── tests/                    # Pruebas unitarias
│   ├── test_api.py
│   ├── test_data_pipeline.py
│   └── test_model_artifacts.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── DATA_PROVENANCE.md
└── README.md
```

---

## 1. Preparación del entorno

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# o
source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

---

## 2. Control de datos y trazabilidad

```bash
python scripts/00_provenance.py
```

Genera:
- `DATA_PROVENANCE.md`: hash y metadata del dataset.
- `data/provenance.json`: trazabilidad reproducible.

---

## 3. Validación e imputación de datos

```bash
python scripts/01_validar_datos.py
```

Este script:
- Valida el esquema (`models/schema.json`).
- Convierte tipos erróneos.
- Imputa valores nulos (mediana / moda según tipo).
- Guarda `data/processed/housing_clean.csv`.

---

## 4. Entrenamiento del modelo base

```bash
python scripts/02_train_baseline.py
```

Modelos evaluados:
- LinearRegression
- Ridge (con GridSearchCV)
- RandomForestRegressor (con GridSearchCV)
- XGBRegressor (con GridSearchCV)

Selecciona automáticamente el modelo con menor RMSE.  
Guarda:
- `models/best_model.pkl`
- `models/preprocessor.pkl`
- `models/metrics.json`
- `models/model_metadata.json`

---

## 5. Retraining automatizado

```bash
python scripts/03_retrain_if_better.py
```

- Compara el modelo actual vs nuevo (por RMSE).  
- Si mejora, reemplaza los artefactos en `models/`.  
- Registra la fecha y métrica en logs.

**Ejemplo de salida:**
```
RMSE nuevo: 2.49 | RMSE actual: 2.50
Nuevo modelo más preciso. Artefactos reemplazados.
```

---

## 6. Detección de Drift

```bash
python scripts/04_drift_check.py
```

- Compara las estadísticas de distribución del dataset actual vs histórico.  
- Genera `logs/drift_report.json`.  
- Si detecta drift significativo, activa el retraining en el pipeline CI/CD.

---

## 7. Pruebas unitarias

```bash
pytest -q
```

Incluye:
- `test_api.py` → valida endpoints `/health` y `/predict`.  
- `test_data_pipeline.py` → asegura limpieza e imputación correctas.  
- `test_model_artifacts.py` → verifica la existencia e integridad de modelos.

---

## 8. API REST (FastAPI)

Archivo: `app/server.py`

Endpoints:
- `GET /health` → verifica disponibilidad de artefactos.  
- `POST /predict` → recibe JSON con features y devuelve predicción.

**Ejemplo de uso:**

```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"CRIM":0.1,"ZN":18,"INDUS":2.31,"CHAS":0,"NOX":0.538,"RM":6.575,"AGE":65.2,"DIS":4.09,"RAD":1,"TAX":296,"PTRATIO":15.3,"B":396.9,"LSTAT":4.98}'
```

**Respuesta esperada:**
```json
{"prediction": 33.39}
```

---

## 9. Docker

**Dockerfile**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY models models
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Construcción y ejecución:**
```bash
docker build -t mlops_boston .
docker run -p 8000:8000 mlops_boston
```

---

## 10. CI/CD en GitHub Actions

Workflow: `.github/workflows/mlops.yml`

Jobs:
1. **build-train-test:** valida, limpia, entrena y corre tests.  
2. **drift-check:** ejecuta `04_drift_check.py` y genera reportes.  
3. **retrain:** reentrena solo si hay drift o `force_retrain=true`.  
4. **docker-image:** siempre construye y publica la imagen en GHCR.

Imagen publicada:  
`ghcr.io/agrisalesa/mlops_boston:latest`

---

## 11. Demo en Codespaces

```bash
docker pull ghcr.io/agrisalesa/mlops_boston:latest
docker run -d -p 8000:8000 ghcr.io/agrisalesa/mlops_boston:latest
```

Abrir en navegador:
```
https://<tu_codespace>.github.dev/docs
```

---

## 12. Próximos pasos

- Integrar Evidently o MLflow para métricas y monitoreo continuo.  
- Exponer versión de modelo vía `/metadata`.  
- Despliegue automático en AWS ECS / Azure App Service / GCP Cloud Run.

---

## Conclusión

Este proyecto demuestra:
- Un flujo MLOps completo y automatizado.  
- Un modelo auto-reentrenable y trazable.  
- Despliegue reproducible mediante Docker + CI/CD.  

Ideal para entornos productivos con bajo mantenimiento manual.
