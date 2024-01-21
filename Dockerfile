FROM python:3.9-slim

WORKDIR /app

COPY . /app/

CMD ["python", "./machine_learning_iris/main.py"]