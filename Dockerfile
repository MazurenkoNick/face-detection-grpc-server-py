# Dockerfile
FROM python:3.9-slim

# Встановлюємо залежності для OpenCV
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean

# Створюємо робочу директорію
WORKDIR /app

# Копіюємо файли проекту
COPY requirements.txt .
COPY . .

# Встановлюємо Python-залежності
RUN pip install --no-cache-dir -r requirements.txt

# Відкриваємо порт для gRPC
EXPOSE 50051

# Запускаємо сервер
CMD ["python", "face_detection_example.py"]
