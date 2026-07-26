FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ libmagic1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Use async-capable gunicorn worker
RUN pip install --no-cache-dir gunicorn uvicorn
EXPOSE 5000
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "120", "-b", "0.0.0.0:5000", "app.api:app"]
