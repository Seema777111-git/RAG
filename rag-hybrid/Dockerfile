# syntax=docker/dockerfile:1
#
# Two small targets built from one file:
#   api : FastAPI backend (FAISS, NetworkX, NeMo Guardrails, DeepEval, sentence-transformers on CPU)
#   ui  : Streamlit front end (only streamlit + httpx, so the image stays light)
#
#   docker build --target api -t rag-hybrid-api .
#   docker build --target ui  -t rag-hybrid-ui  .
# Normally you just run:  docker compose up --build

FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app:app /app


# ------------------------------------------------------------------------------------ API
FROM base AS api
# libgomp1: OpenMP runtime used by FAISS / PyTorch on slim images.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only PyTorch first (the default Linux wheel bundles several GB of CUDA libraries we do not need).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY backend ./backend

# Models downloaded from Hugging Face are cached in a volume so they survive rebuilds.
ENV HF_HOME=/hf-cache \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES
RUN mkdir -p /hf-cache && chown app:app /hf-cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)" || exit 1
CMD ["python", "-m", "backend.app"]


# ------------------------------------------------------------------------------------- UI
FROM base AS ui
COPY requirements-ui.txt .
RUN pip install -r requirements-ui.txt

COPY frontend ./frontend
COPY .streamlit ./.streamlit
USER app

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1
CMD ["python", "-m", "streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
