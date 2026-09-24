FROM python:3.12-slim

WORKDIR /app

COPY requirements/base.txt requirements/base.txt
COPY requirements/dev.txt requirements/dev.txt

RUN pip install --no-cache-dir -r requirements/dev.txt \
    && pip install --no-cache-dir fastapi uvicorn

COPY app app
COPY src src
COPY config config
COPY models models

# Only the results JSON is needed for MLflow registration.
# No .ipynb notebooks are copied into the image.
COPY notebooks/results_summary.json notebooks/results_summary.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]