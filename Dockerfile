# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Installs all Python dependencies into a venv so build tools
# (gcc, g++, etc.) don't pollute the final image.
FROM python:3.11-slim AS builder

WORKDIR /app

# Build-time system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Install CPU-only torch first (avoids pulling the full CUDA wheel ~2.5 GB).
# The --index-url flag must come before -r so the custom index applies to torch/torchvision.
RUN pip install --no-cache-dir \
    "torch>=2.2.0" "torchvision" \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime system deps only (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from builder (no compiler toolchain in final image)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Persistent data directory — overridden on Render by a disk mount at /mnt/data
ENV DATA_PERSIST_DIR=/mnt/data
RUN mkdir -p /mnt/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
