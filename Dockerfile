FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The model bundle is built by train_churn.py; copy it in at deploy time if
# it is not baked into the image (e.g. via a build stage or volume).
RUN python -c "import os; os.makedirs('data/store', exist_ok=True)"

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
