FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb

WORKDIR /app

COPY backend/requirement.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirement.txt

COPY backend/ .

RUN mkdir -p /app/uploads /app/logs /app/models

EXPOSE 8000

CMD ["sh", "-c", "python download_models.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
