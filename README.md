# Proyecto MLOps — Boston Housing

Autor: Andrés Grisales Ardila

Implementación completa de un flujo MLOps para un modelo de regresión con el dataset Boston Housing, incluyendo:
- Control de versiones y trazabilidad de datos.
- Validación e imputación robusta.
- Entrenamiento con múltiples modelos y optimización.
- Selección automática del mejor modelo.
- Servir el modelo mediante una API Flask.
- Contenerización con Docker y automatización en GitHub Actions (CI/CD).

---

## Estructura del proyecto

```
mlops_boston/
├── .github/workflows/        # Pipeline CI/CD (mlops.yml)
├── app/                      # API Flask (server.py)
├── data/                     # Datos crudos y procesados
├── logs/                     # Logs del proceso
├── models/                   # Modelos, preprocesador y metadatos
├── scripts/                  # Scripts principales del flujo
│   ├── 00_provenance.py
│   ├── 01_validar_datos.py
│   └── 02_train_baseline.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── DATA_PROVENANCE.md
└── README.md
```

---

## 1. Preparación del entorno

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

---

## 2. Control de datos y trazabilidad

Ejecutar:
```bash
python scripts/00_provenance.py
```

Genera:
- `DATA_PROVENANCE.md`: documento con hash y detalles del dataset.  
- `data/provenance.json`: trazabilidad del archivo crudo.

Esto asegura que el dataset usado para entrenamiento sea el mismo que se valida y se despliega.

---

## 3. Validación e imputación de datos

Ejecutar:
```bash
python scripts/01_validar_datos.py
```

Este script:
- Valida la integridad del archivo (SHA256).
- Corrige tipos de datos según esquema.
- Imputa valores faltantes con lógica robusta:
  - Floats / int no binarios → mediana.
  - Variables binarias (0/1) → moda.
- Genera `data/processed/housing_clean.csv`.

---

## 4. Entrenamiento del modelo

Ejecutar:
```bash
python scripts/02_train_baseline.py
```

Modelos usados:
- LinearRegression
- Ridge (GridSearchCV)
- RandomForestRegressor (GridSearchCV)
- XGBRegressor (GridSearchCV)

Selecciona el modelo con menor RMSE y guarda:
- `models/best_model.pkl`
- `models/preprocessor.pkl`
- `models/metrics.json`
- `models/model_metadata.json`

Salida esperada (ejemplo):
```
Métricas (test):
  linear_regression -> RMSE: 5.00
  random_forest    -> RMSE: 3.28
  xgboost          -> RMSE: 2.36
Mejor modelo: xgboost
Artefactos guardados en 'models/'
```

---

## 5. API Flask

Archivo: `app/server.py`

Endpoints:
- GET /health → verifica artefactos (best_model, preprocessor, schema).
- POST /predict → recibe JSON y devuelve predicción.

Ejemplo de uso:

```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"CRIM":0.1,"ZN":18,"INDUS":2.31,"CHAS":0,"NOX":0.538,"RM":6.575,"AGE":65.2,"DIS":4.09,"RAD":1,"TAX":296,"PTRATIO":15.3,"B":396.9,"LSTAT":4.98}'
```

Respuesta esperada:
```json
{"prediction": 33.39}
```

---

## 6. Docker

**Dockerfile:**

```dockerfile
FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY models models

EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app.server:app"]
```

**.dockerignore:**
```
__pycache__/
*.pyc
*.log
.venv/
data/
logs/
```

Construcción y ejecución:
```bash
docker build -t mlops_boston .
docker run -p 8000:8000 mlops_boston
```

---

## 7. CI/CD con GitHub Actions

Workflow `.github/workflows/mlops.yml`:

- Entrenamiento automático del modelo en la rama principal.  
- Construcción y publicación de la imagen en GitHub Container Registry (GHCR).

Imagen generada:  
`ghcr.io/agrisalesa/mlops_boston:latest`

---

## 8. Despliegue en Codespaces

1. Abre el Codespace del repositorio.  
2. En terminal, ejecuta:
   ```bash
   docker pull ghcr.io/agrisalesa/mlops_boston:latest
   docker run -d -p 8000:8000 ghcr.io/agrisalesa/mlops_boston:latest
   ```
3. En el panel Ports, haz público el puerto 8000.  
4. Abre en navegador:  
   `https://<tu_codespace>.github.dev/health`

---

## 9. Próximos pasos

- Migrar a FastAPI para documentación interactiva.  
- Agregar monitoreo de drift y performance con MLflow o Evidently.  
- Desplegar en AWS, Azure o GCP.

---

Repositorio: [https://github.com/agrisalesa/mlops_boston](https://github.com/agrisalesa/mlops_boston)  
Imagen Docker: `ghcr.io/agrisalesa/mlops_boston:latest`
