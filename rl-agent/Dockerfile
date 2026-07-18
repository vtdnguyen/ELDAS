# syntax=docker/dockerfile:1
# (Dòng syntax trên BẮT BUỘC để dùng `RUN --mount=type=cache` bên dưới.)

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
# BuildKit pip cache mount: các wheel đã tải (torch/scipy/... ~1GB) được GIỮ LẠI
# giữa các lần build — kể cả khi layer bị invalidate (đổi 1 dòng requirements)
# HOẶC khi Docker evict layer-cache. Bỏ --no-cache-dir để pip GHI vào cache đó.
# ⇒ lần build đầu vẫn tải; MỌI lần sau reuse wheel, gần như tức thời (không phụ
#   thuộc mạng). Cache mount không nằm trong image nên KHÔNG làm image to thêm.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

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