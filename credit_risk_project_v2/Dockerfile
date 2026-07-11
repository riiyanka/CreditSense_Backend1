FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# artifacts/ must contain the trained .joblib files (run train.py first,
# or copy them in — do NOT re-train inside the image, it's slower and
# the historical dataset shouldn't ship inside your API container)
COPY artifacts/ ./artifacts/
COPY app/ ./app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
