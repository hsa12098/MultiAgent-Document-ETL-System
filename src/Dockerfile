FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# system deps for pdf parsing and tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# copy code
COPY . /app

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src/app:app", "--host", "0.0.0.0", "--port", "8000"]
