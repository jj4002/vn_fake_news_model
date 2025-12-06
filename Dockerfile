FROM python:3.11.9

# Env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 \
    KMP_AFFINITY=granularity \
    KMP_BLOCKTIME=0

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Create runtime dirs (trước khi copy code)
RUN mkdir -p models temp data

# Copy ONLY requirements first (để cache pip install)
COPY requirement.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt

# Update yt-dlp to latest (TikTok extractor changes frequently)
RUN pip install --no-cache-dir -U yt-dlp

# Copy app code (để sau cùng, khi code thay đổi không ảnh hưởng pip install)
COPY . .

# Create non-root user and set ownership
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
