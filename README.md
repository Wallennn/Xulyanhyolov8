# Phát Hiện Vật Thể & Xử Lý Ảnh với YOLOv8

Ứng dụng thị giác máy tính thông minh sử dụng **YOLOv8** kết hợp nhận diện văn bản tiếng Việt (**Tesseract OCR**), phân tích màu sắc và giao diện điều khiển trực quan (**Tkinter**).

---

## 🌟 Các tính năng chính

- **Phát hiện vật thể thời gian thực**: Sử dụng mô hình YOLOv8 (`yolov8s.pt`, hỗ trợ chuyển đổi linh hoạt sang các phiên bản khác).
- **Giao diện người dùng trực quan (GUI)**:
  - Xem trực tiếp từ Camera, file Video hoặc xử lý file Ảnh.
  - Hỗ trợ chế độ giao diện Sáng / Tối (Light / Dark mode).
  - Tự động Việt hóa tên nhãn (có dấu hoặc không dấu).
- **Nhận diện màu sắc & Văn bản (OCR)**:
  - Phân tích màu sắc chủ đạo của vật thể được phát hiện.
  - Tích hợp Tesseract OCR nhận diện chữ trên khung hình.
- **Thống kê & Cảnh báo**:
  - Đếm số lượng vật thể theo thời gian thực và hiển thị biểu đồ Matplotlib trực tiếp.
  - Phát âm thanh cảnh báo khi phát hiện vật thể mục tiêu được cấu hình.
  - Tự động chụp liên tiếp khi số lượng vật thể vượt ngưỡng quy định.
  - Xuất báo cáo dữ liệu thống kê ra file định dạng CSV.
  - Ghi và lưu video phát hiện.

---

## ⌨️ Phím tắt nhanh

| Phím tắt | Chức năng |
| :---: | :--- |
| `C` | Mở / Đóng Camera |
| `S` | Lưu ảnh chụp khung hình hiện tại |
| `P` | Tạm dừng / Tiếp tục |
| `R` | Đặt lại chương trình về ban đầu |

---

## 🚀 Hướng dẫn cài đặt & Chạy ứng dụng

### Cách 1: Chạy trực tiếp trên máy cục bộ (Local)

#### 1. Yêu cầu tiên quyết
- Python 3.8+ trở lên.
- Đã cài đặt [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) (khuyến nghị cho Windows cài tại `C:\Program Files\Tesseract-OCR`).

#### 2. Cài đặt thư viện
```bash
# Clone repository
git clone https://github.com/Wallennn/Xulyanhyolov8.git
cd Xulyanhyolov8

# Cài đặt các dependencies
pip install -r requirements.txt
```

#### 3. Khởi chạy ứng dụng
```bash
python phathienvatthe.py
```

---

### Cách 2: Chạy bằng Docker / Docker Compose

Dự án đã được cấu hình đầy đủ `Dockerfile` và `docker-compose.yml` tích hợp sẵn Tesseract OCR, Tkinter và OpenCV.

> **Lưu ý về hiển thị GUI từ Docker Container:**
> - **Trên Linux:** Cấp quyền hiển thị X11:
>   ```bash
>   xhost +local:root
>   ```
> - **Trên Windows:** Cần một X-Server (ví dụ [VcXsrv](https://sourceforge.net/projects/vcxsrv/)):
>   1. Mở XLaunch, chọn: *Multiple windows* -> *Start no client* -> tích chọn *Disable access control*.
>   2. Địa chỉ `DISPLAY` mặc định trong compose đã được trỏ tới `host.docker.internal:0.0`.

#### Sử dụng Docker Compose:
```bash
# Build và khởi chạy container
docker compose up --build

# Dừng container
docker compose down
```

#### Sử dụng Docker CLI:
```bash
# 1. Build image
docker build -t xulyanh-yolov8 .

# 2. Chạy container (Ví dụ trên Linux)
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device=/dev/video0:/dev/video0 \
  xulyanh-yolov8
```

---

## 📁 Cấu trúc thư mục

```text
├── phathienvatthe.py    # Mã nguồn chính của ứng dụng
├── config.json          # File cấu hình mặc định (ngưỡng tin cậy, giao diện,...)
├── yolov8s.pt           # Trọng số mô hình YOLOv8s
├── requirements.txt     # Danh sách thư viện Python cần thiết
├── Dockerfile           # Cấu hình đóng gói Docker container
├── docker-compose.yml   # Khởi chạy nhanh với Docker Compose
├── .dockerignore        # Bỏ qua các file không cần thiết khi build Docker
├── .gitignore           # Bỏ qua các file tạm trong Git
└── README.md            # Tài liệu hướng dẫn sử dụng
```

---

## ⚙️ Cấu hình tùy chỉnh (`config.json`)

Bạn có thể thay đổi các thông số mặc định của ứng dụng trực tiếp trên giao diện hoặc chỉnh sửa file `config.json`:
```json
{
    "label_style": "Không dấu",
    "model": "yolov8s.pt",
    "confidence": 0.44,
    "alert_object": "person",
    "resize_scale": 1.2,
    "theme": "light",
    "auto_capture_object": "apple",
    "auto_capture_threshold": 3
}
```
