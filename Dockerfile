FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps: ffmpeg for audio conversion; git for senko install; basic build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    ca-certificates \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN pip install av

RUN apt-get update && apt-get install -y curl

RUN apt-get update && apt-get install -y wget
RUN apt-get update && apt-get install -y tar

RUN pip install python-multipart
RUN pip install librosa

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

