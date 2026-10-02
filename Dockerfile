# Imagen para Google Cloud Run. La app no descarga nada en tiempo de ejecución:
# todos los datos están en data/ como CSV versionados.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit ./.streamlit
COPY app.py ./
COPY src ./src
COPY data ./data

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

# Cloud Run inyecta $PORT; Streamlit debe escuchar en 0.0.0.0
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
