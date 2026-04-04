# Stage 1: BUILD
# Dùng image đầy đủ để cài các thư viện cần compile (numpy, etc.)
FROM python:3.11.9-slim-bookworm AS builder

# Cài build tools — một số thư viện Python cần gcc để compile
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements trước — tận dụng Docker cache layer
# Giống trick pom.xml: nếu requirements.txt không đổi → không reinstall
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: RUNTIME
FROM python:3.11.9-slim-bookworm AS runtime

WORKDIR /app

# Copy toàn bộ thư viện đã cài từ stage builder vào đúng chỗ Python tìm
COPY --from=builder /install /usr/local

# Copy source code
COPY src ./src

# Tạo thư mục output
RUN mkdir -p /data/results /data/models

# Không chạy root
RUN groupadd --gid 1001 rluser \
    && useradd --uid 1001 --gid rluser --shell /bin/bash --create-home rluser \
    && chown -R rluser:rluser /app /data

USER rluser

# Không expose port vì rl-agent là client, nó GỌI ra Java chứ không lắng nghe
CMD ["python", "src/train.py"]