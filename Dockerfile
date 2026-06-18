FROM python:3.12.3-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip==24.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code AND the HTML frontend
COPY app/ ./app/
COPY index.html ./

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
