FROM python:3.10-slim

WORKDIR /app

COPY MLProject/requirements.txt MLProject/requirements.txt

RUN pip install no-cache-dir -r MLProject/requirements.txt

EXPOSE 5000

CMD ["mlflow","server","--host","0.0.0.0","--port","5000","--backend-store-uri","sqlite:///mlruns.db","--default-artifact-root","file:/app/mlruns"]