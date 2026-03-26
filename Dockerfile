FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

ENV PYTHONPATH=/app/src

RUN pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "podmemory.main:app", "--host", "0.0.0.0", "--port", "8000"]
