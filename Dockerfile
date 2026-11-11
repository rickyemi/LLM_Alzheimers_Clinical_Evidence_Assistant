# syntax=docker/dockerfile:1
# =============================================================================
# AD Clinical Evidence Assistant: multi-stage image.
#   docker build -t ad-rag:0.1.0 .
#   docker build --build-arg EXTRAS=st -t ad-rag:st .     # + sentence-transformers
# Secrets (ANTHROPIC_API_KEY, API_KEY) are passed at RUNTIME, never baked in.
# =============================================================================

# ---- Stage 1: build wheels so the runtime image has no compiler/build tools ----
FROM python:3.11-slim AS builder
ARG EXTRAS=""
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir build \
 && if [ -n "$EXTRAS" ]; then SPEC=".[${EXTRAS}]"; else SPEC="."; fi \
 && pip wheel --no-cache-dir --wheel-dir /wheels "$SPEC"

# ---- Stage 2: slim runtime ----
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    AD_RAG_HOME=/app PORT=8000
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Config, corpus and eval data are part of the release (versioned together).
COPY configs ./configs
COPY data ./data
COPY scripts ./scripts

# Non-root user; only artifacts/ is writable (index, audit log, review queue).
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/artifacts && chown -R appuser:appuser /app/artifacts
USER appuser

# Build the index at image build time so the container starts instantly and the
# image carries a fixed corpus_version. Re-built automatically if missing at runtime.
RUN python scripts/build_index.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "ad_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
