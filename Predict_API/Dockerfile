FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

ENV GOOGLE_APPLICATION_CREDENTIALS=ecoquest-vertex-key.json

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
