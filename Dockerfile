FROM python:3.10.13-slim

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "--workers=1", "--bind", "0.0.0.0:8080", "--worker-class", "uvicorn.workers.UvicornWorker", "main:app", "--access-logfile", "-"]