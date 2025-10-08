# 🧠 Proyecto MLOps — Boston Housing

**Autor:** Andrés Grisales Ardila  

**Descripción:**  
Implementación completa de un flujo MLOps de extremo a extremo para un modelo de regresión usando el dataset *Boston Housing*.  
Incluye trazabilidad de datos, validación robusta, entrenamiento con múltiples modelos, selección automática del mejor modelo, despliegue mediante una API Flask, y pipeline automatizado de CI/CD con GitHub Actions y Docker.

---

## 🧩 Estructura del proyecto

```
mlops_boston/
│
├── .github/workflows/          # Pipeline CI/CD (mlops.yml)
├── app/                        # API Flask (endpoints health y predict)
├── data/                       # Datos crudos, procesados y metadatos
├── logs/                       # Registros de ejecución y validación
├── models/                     # Modelos y preprocesadores entrenados
├── scripts/                    # Entrenamiento, validación y pruebas
│
├── .dockerignore               # Archivos y carpetas ignoradas por Docker
├── Dockerfile                  # Imagen de despliegue (Flask + modelo)
├── DATA_PROVENANCE.md          # Hash SHA256 del dataset original
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación principal del proyecto
```

---

## ⚙️ 1. Entorno y dependencias

Crear un entorno virtual e instalar dependencias:

```bash
python -m venv .venv
source .venv/bin/activate    # En Linux/Mac
.venv\Scripts\activate     # En Windows

pip install -r requirements.txt
```

---

## 🧠 2. Entrenamiento del modelo

Ejecutar el pipeline de entrenamiento (ejemplo desde `scripts/train.py`):

```bash
python scripts/train.py
```

Esto genera:
- `models/best_model.pkl` → modelo entrenado y optimizado  
- `models/preprocessor.pkl` → transformaciones de datos  
- Logs de validación y métricas en `/logs`  

---

## 🌍 3. API Flask — Servidor del modelo

El archivo principal `app/main.py` define los **endpoints**:

```python
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "best_model": True,
        "preprocessor": True,
        "schema": True,
        "status": "ok"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    X = preprocessor.transform(df)
    prediction = model.predict(X)
    return jsonify({"prediction": float(prediction[0])})
```

### 🔹 Ejecutar localmente

```bash
python app/main.py
```

Verifica los endpoints:

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/predict` con JSON de entrada.

Ejemplo de cuerpo JSON:
```json
{
  "CRIM": 0.1,
  "ZN": 18,
  "INDUS": 2.31,
  "CHAS": 0,
  "NOX": 0.538,
  "RM": 6.575,
  "AGE": 65.2,
  "DIS": 4.09,
  "RAD": 1,
  "TAX": 296,
  "PTRATIO": 15.3,
  "B": 396.9,
  "LSTAT": 4.98
}
```

---

## 🐳 4. Despliegue con Docker

**Dockerfile:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app.main:app"]
```

**.dockerignore:**
```
__pycache__/
*.pyc
*.pkl
*.log
.venv/
data/
logs/
```

Construir y ejecutar la imagen:

```bash
docker build -t mlops_boston .
docker run -p 8000:8000 mlops_boston
```

---

## ⚙️ 5. CI/CD con GitHub Actions

Archivo: `.github/workflows/mlops.yml`

```yaml
name: mlops-ci

on:
  push:
    branches: [ "main" ]

jobs:
  build-train-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: python scripts/train.py

      - name: Save model artifact
        uses: actions/upload-artifact@v3
        with:
          name: model-artifacts
          path: models/

  docker:
    needs: build-train-test
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v3
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
```

---

## 🚀 6. Despliegue en GitHub Codespaces

Puedes ejecutar el contenedor directamente en tu entorno Codespaces.  
El puerto **8000** debe marcarse como **“Public”** y abrirlo desde el panel *Ports*.

### Para probar:

- Endpoint de salud:  
  `GET https://<tu_codespace>.github.dev/health`  
- Endpoint de predicción:  
  `POST https://<tu_codespace>.github.dev/predict`

---

## 🔍 7. Conceptos clave aprendidos

| Concepto | Explicación |
|-----------|-------------|
| **Puerto 8000** | Es la “puerta” por donde Flask escucha las peticiones HTTP. |
| **Endpoint** | Es la “ruta” o función específica del servidor que responde a una URL. |
| **Flask** | Microframework de Python que convierte funciones en servicios web. |
| **Docker** | Empaqueta el proyecto con todas sus dependencias en un contenedor reproducible. |
| **GitHub Actions** | Automatiza el entrenamiento, pruebas y publicación del contenedor. |

---

## 🧭 8. Flujo general del proyecto

1️⃣ Validación y preprocesamiento de datos  
2️⃣ Entrenamiento con múltiples modelos  
3️⃣ Selección del mejor modelo  
4️⃣ Serialización (`joblib`) de modelo y preprocesador  
5️⃣ Despliegue con Flask  
6️⃣ Contenerización con Docker  
7️⃣ Automatización con GitHub Actions  
8️⃣ Ejecución y prueba en Codespaces  

---

## 📈 9. Ejemplo de flujo en producción (visual)

El flujo de peticiones hacia la API Flask:

![Diagrama de flujo API Flask](A_diagram_in_a_digital_vector_graphic_format_illus.png)

---

## 🏁 10. Próximos pasos sugeridos

- Migrar de **Flask** a **FastAPI** para documentación automática.  
- Incluir monitoreo de predicciones con MLflow o EvidentlyAI.  
- Desplegar en un entorno gestionado (AWS, Azure o Google Cloud).  

---

📌 **Repositorio:** [https://github.com/agrisalesa/mlops_boston](https://github.com/agrisalesa/mlops_boston)  
📦 **Imagen Docker:** `ghcr.io/agrisalesa/mlops_boston:latest`  
