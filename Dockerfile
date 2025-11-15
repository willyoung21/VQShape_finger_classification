# Imagen base
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Directorio principal
WORKDIR /app

# Instala dependencias
COPY app/requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# Copia el contenido completo del proyecto
COPY VQShape/ /app/VQShape/
COPY app/app.py /app/

# 🔹 Añade esta línea: permite que Python vea /app/VQShape como módulo raíz
ENV PYTHONPATH="/app/VQShape:${PYTHONPATH}"

# Puerto de streamlit
EXPOSE 8501

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
