FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY best_model.pkl .

EXPOSE 8000

CMD ["python", "predict.py"]
