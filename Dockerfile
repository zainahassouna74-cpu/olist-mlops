FROM python:3.12-slim

WORKDIR /app

COPY requirements/base.txt requirements/base.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements/base.txt

COPY app app
COPY src src
COPY config config

COPY models models
COPY notebooks/results_summary.json notebooks/results_summary.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]