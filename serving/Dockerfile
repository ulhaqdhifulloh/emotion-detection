FROM python:3.10-slim

# 1) OS deps minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libjpeg62-turbo-dev libpng-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) Workdir & install pip deps
WORKDIR /app
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy code
COPY serving /app/serving
COPY src /app/src
# (Opsional) Copy model saat build; atau mount via volume di runtime.
# Pastikan file ini ada, jika tidak, hapus baris di bawah dan pasang via volume/env.
COPY checkpoints/best.pt /app/best.pt

# 4) Env & CMD
ENV CKPT_PATH=/app/best.pt \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8080
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]