FROM python:3.10-slim

WORKDIR /app

COPY MLProject/requirements.txt MLProject/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

EXPOSE 5000

CMD ["python","MLProject/modelling.py"]