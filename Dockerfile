FROM python:3.10-slim

WORKDIR /app

COPY MLProject/ .

RUN pip install --no-cache-dir -r MLProject/requirements.txt

EXPOSE 5000

CMD ["python","MLProject/modelling.py"]