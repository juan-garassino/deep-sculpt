FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
# torch matches the version the TorchScript files were exported with (export.py, deep-sculpt .venv)
RUN pip install --no-cache-dir uv==0.13.5 \
 && uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
 && uv pip install --system --no-cache numpy==1.26.4 fastapi==0.116.1 "uvicorn[standard]==0.53.0" pydantic==2.13.5
WORKDIR /app
COPY app.py /app/app.py
COPY weights/ /app/weights/
ENV DEEPSCULPT_WEIGHTS=/app/weights PORT=8080
RUN useradd -m -u 10001 sculpt
USER sculpt
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health',timeout=3)"
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
