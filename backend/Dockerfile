# Usa una imagen base ligera con Python 3.11
FROM python:3.11-slim

# Define el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de requisitos (si los tienes)
COPY requirements.txt .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código del backend
COPY . .

# Expón el puerto interno 5000 (para Traefik)
EXPOSE 5000

# Comando por defecto: iniciar FastAPI (ajustaremos esto más adelante)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
