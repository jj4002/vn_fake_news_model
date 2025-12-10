# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies cho build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (đã XÓA dòng torch ra khỏi file này)
COPY requirement.txt .

# Cài PyTorch CPU-only TRƯỚC (nhẹ hơn ~3GB so với CUDA)
RUN pip install --no-cache-dir --user \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Cài các dependencies còn lại
RUN pip install --no-cache-dir --user -r requirement.txt

# Cài yt-dlp bản thường (không dùng nightly)
RUN pip install --no-cache-dir --user -U yt-dlp

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Cài runtime dependencies tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages từ builder
COPY --from=builder /root/.local /root/.local

# Copy source code
COPY . .

# Set PATH
ENV PATH=/root/.local/bin:$PATH

# Tạo thư mục cần thiết (bỏ data vì bạn đã xóa)
RUN mkdir -p /app/models /app/temp

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
