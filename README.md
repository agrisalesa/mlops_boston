# 🧠 Proyecto MLOps — Boston Housing
**Autor:** Andrés Grisales Ardila  
**Descripción:** Implementación completa de un flujo MLOps de extremo a extremo para un modelo de regresión usando el dataset Boston Housing.  
Incluye trazabilidad de datos, validación robusta, entrenamiento con múltiples modelos, selección automática del mejor modelo y despliegue mediante una API Flask.

---

## 📁 Estructura del proyecto

```
mlops_boston/
│
├── .venv/                  # Entorno virtual de Python
├── app/                    # API Flask (endpoints health y predict)
├── data/                   # Datos crudos, procesados y metadatos
├── logs/                   # Registros de inferencias
├── models/                 # Artefactos del modelo y metadatos
├── scripts/                # Scripts de procesamiento, validación y entrenamiento
│
├── DATA_PROVENANCE         # Archivo con el hash SHA256 del dataset
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación principal del proyecto
```

---

## ⚙️ Instalación del entorno

### 1. Crear y activar entorno virtual

**Windows CMD**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Mac / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

Crea un archivo `requirements.txt` en la raíz con:

```
numpy
pandas
scikit-learn
joblib
xgboost
flask
```

Instala todo:
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Verifica instalación:
```bash
python -c "import flask, joblib, xgboost, sklearn, pandas, numpy; print('OK')"
```

---

## 📊 Procesamiento de datos

Coloca el dataset original en:
```
data/raw/housing.csv
```

### 1. Trazabilidad y hash
Calcula el hash SHA256 del dataset y guarda la información en `provenance.json`:
```bash
python scripts/00_provenance.py
```

Genera:
- `DATA_PROVENANCE.md`
- `data/provenance.json`

### 2. Validación, imputación y limpieza
Ejecuta:
```bash
python scripts/01_validar_datos.py
```

Acciones que realiza:
- Normaliza columnas y tipos.
- Verifica contrato vs `schema.json`.
- Imputa valores nulos:
  - Binarias (ej. `CHAS`) → moda  
  - Índices (ej. `RAD`) → mediana redondeada  
  - Resto de numéricas → mediana  
- Elimina filas con NaN en la variable objetivo `MEDV`.  
- Guarda el dataset limpio en:
  ```
  data/processed/housing_clean.csv
  ```

---

## 🧩 Entrenamiento del modelo

Entrena y selecciona automáticamente el mejor modelo según RMSE:
```bash
python scripts/02_train_baseline.py
```

Modelos incluidos:
- **Linear Regression**
- **Ridge Regression** (grid de α)
- **Random Forest Regressor** (grid)
- **XGBoost Regressor** (grid)

Solo se guarda el mejor modelo.

Archivos generados:
```
models/
 ├── best_model.pkl
 ├── preprocessor.pkl
 ├── metrics.json
 └── model_metadata.json
```

---

## 🚀 API Flask — Servir el modelo

Inicia el servidor:
```bash
python app/server.py
```

Por defecto:
```
http://127.0.0.1:8000
```

### 1. Verificar estado
```bash
curl http://127.0.0.1:8000/health
```

Respuesta esperada:
```json
{
  "status": "ok",
  "preprocessor": true,
  "best_model": true,
  "schema": true
}
```

### 2. Hacer una predicción

**Windows CMD (una línea):**
```bat
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"CRIM\":0.1,\"ZN\":18,\"INDUS\":2.31,\"CHAS\":0,\"NOX\":0.538,\"RM\":6.575,\"AGE\":65.2,\"DIS\":4.09,\"RAD\":1,\"TAX\":296,\"PTRATIO\":15.3,\"B\":396.9,\"LSTAT\":4.98}"
```

**Windows PowerShell:**
```powershell
$body = @{
  CRIM=0.1; ZN=18; INDUS=2.31; CHAS=0; NOX=0.538; RM=6.575;
  AGE=65.2; DIS=4.09; RAD=1; TAX=296; PTRATIO=15.3; B=396.9; LSTAT=4.98
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
```

**Mac / Linux (curl):**
```bash
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"CRIM":0.1,"ZN":18,"INDUS":2.31,"CHAS":0,"NOX":0.538,"RM":6.575,"AGE":65.2,"DIS":4.09,"RAD":1,"TAX":296,"PTRATIO":15.3,"B":396.9,"LSTAT":4.98}'
```

Ejemplo de respuesta:
```json
{"prediction": 27.83}
```

---

## ✅ Validación del payload

El endpoint `/predict` valida:
- Estructura completa (todas las columnas del `schema`).
- Tipos (`int` o `float`).
- Rango razonable por variable (`RANGE_RULES` en `server.py`).
- Reglas específicas:
  - `CHAS` ∈ {0,1}  
  - `RAD` entero positivo

Si hay errores:
```json
{
  "error": "CHAS: debe ser 0 o 1; RAD: debe ser entero positivo (>=1)"
}
```
(HTTP 422)

---

## 🧾 Logging

Cada inferencia se guarda en:
```
logs/predictions.csv
```

Con columnas:
```
CRIM,ZN,INDUS,CHAS,...,LSTAT,prediction,ts
```

Esto permite trazabilidad de cada request.

---

## 🧠 Recomendaciones finales

- Para producción, usar un servidor WSGI:
  ```bash
  pip install gunicorn
  gunicorn -w 2 -b 0.0.0.0:8000 app.server:app
  ```
- Si el puerto 8000 está ocupado, cámbialo en `server.py`:
  ```python
  app.run(host="127.0.0.1", port=5000, debug=False)
  ```
- El modo `debug=True` es solo para desarrollo (muestra errores y recarga automática).

---

## 🧩 Solución de errores comunes

| Error | Solución |
|-------|-----------|
| `ModuleNotFoundError: No module named 'flask'` | Instala Flask: `pip install flask` |
| `ModuleNotFoundError: No module named 'xgboost'` | Instala XGBoost: `pip install xgboost` |
| `The server stays running` | Es normal, está escuchando peticiones. Abre otra terminal para usar curl. |
| `422 - error de validación` | Alguna variable fuera de rango o tipo incorrecto. Revisa mensaje devuelto. |
| `OSError: [Errno 98] Address already in use` | Cambia el puerto (por ejemplo a 5000). |

---

## 📜 Licencia y uso

Este proyecto es educativo y utiliza el dataset **Boston Housing** con fines de práctica.  
Ajusta el `README`, `schema.json` y las reglas de validación a tus propias políticas o dominio si lo reutilizas para un caso empresarial.
