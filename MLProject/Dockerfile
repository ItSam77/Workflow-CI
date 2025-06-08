FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["conda", "run", "-n", "mlflow-env", "uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
