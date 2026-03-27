FROM python:3.11

WORKDIR /deployment/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variable placeholder (will override in deployment)
ENV MODEL_URL="https://dwmnbrwhjoscbnvwybkk.supabase.co/storage/v1/object/public/models/"

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "deployment.app:app", "--bind", "0.0.0.0:8000", "--workers", "2"]