# Sử dụng image Python 3.10 slim chính thức
FROM python:3.10-slim

# Thiết lập các biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Cài đặt các thư viện hệ thống cần thiết cho GUI (Tkinter), OpenCV, và Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    tesseract-ocr \
    tesseract-ocr-vie \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc
WORKDIR /app

# Cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn và mô hình YOLO
COPY . .

# Lệnh khởi chạy ứng dụng
CMD ["python", "phathienvatthe.py"]
