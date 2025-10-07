# Imagen base liviana
FROM python:3.10-slim

# Evita archivos pyc y buffers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Directorio de trabajo
WORKDIR /app

# Copiar e instalar dependencias
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install gunicorn

# Copiar aplicación y artefactos del modelo
COPY app /app/app
COPY models /app/models

# Exponer el puerto de la API
EXPOSE 8000

# Comando de ejecución con Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app.server:app"]
