import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import time
from collections import defaultdict
from ultralytics import YOLO
import string
import pytesseract
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import shutil
from tkinter import font

# Winsound chỉ hỗ trợ trên Windows
try:
    import winsound
except ImportError:
    winsound = None

# Cấu hình Tesseract OCR
if os.name == 'nt':
    os.environ.setdefault('TESSDATA_PREFIX', r'C:\Program Files\Tesseract-OCR')
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract\tesseract.exe'
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    else:
        print("Lỗi: Không tìm thấy Tesseract trên Windows. Vui lòng cài đặt và cấu hình đường dẫn!")
        pytesseract.pytesseract.tesseract_cmd = None
else:
    # Môi trường Linux / Docker
    tess_path = shutil.which("tesseract")
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path
    else:
        pytesseract.pytesseract.tesseract_cmd = "tesseract"

# Tải mô hình YOLOv8
try:
    model = YOLO("yolov8s.pt")
except Exception as e:
    print(f"Lỗi tải mô hình YOLOv8: {e}")
    model = None

# Từ điển Việt hóa nhãn
vietnamese_labels = {
    "person": ["Người", "Nguoi"], 
    "bicycle": ["Xe đạp", "Xe dap"], 
    "car": ["Xe hơi", "Xe hoi"], 
    "motorbike": ["Xe máy", "Xe may"],
    "aeroplane": ["Máy bay", "May bay"], 
    "bus": ["Xe buýt", "Xe buyt"], 
    "train": ["Tàu hỏa", "Tau hoa"], 
    "truck": ["Xe tải", "Xe tai"],
    "boat": ["Thuyền", "Thuyen"], 
    "traffic light": ["Đèn giao thông", "Den giao thong"], 
    "fire hydrant": ["Trụ cứu hỏa", "Tru cuu hoa"],
    "stop sign": ["Biển dừng", "Bien dung"], 
    "parking meter": ["Đồng hồ đỗ xe", "Dong ho do xe"], 
    "bench": ["Ghế dài", "Ghe dai"],
    "cat": ["Mèo", "Meo"], 
    "dog": ["Chó", "Cho"], 
    "horse": ["Ngựa", "Ngua"], 
    "sheep": ["Cừu", "Cuu"], 
    "cow": ["Bò", "Bo"],
    "elephant": ["Voi", "Voi"], 
    "bear": ["Gấu", "Gau"], 
    "zebra": ["Ngựa vằn", "Ngua van"], 
    "giraffe": ["Hươu cao cổ", "Huou cao co"],
    "backpack": ["Ba lô", "Ba lo"], 
    "umbrella": ["Ô", "O"], 
    "handbag": ["Túi xách", "Tui xach"], 
    "tie": ["Cà vạt", "Ca vat"],
    "suitcase": ["Va li", "Va li"], 
    "frisbee": ["Đĩa bay", "Dia bay"], 
    "skis": ["Ván trượt tuyết", "Van truot tuyet"], 
    "snowboard": ["Ván trượt tuyết", "Van truot tuyet"],
    "sports ball": ["Bóng thể thao", "Bong the thao"], 
    "kite": ["Diều", "Dieu"], 
    "baseball bat": ["Gậy bóng chày", "Gay bong chay"],
    "baseball glove": ["Găng bóng chày", "Gang bong chay"], 
    "skateboard": ["Ván trượt", "Van truot"], 
    "surfboard": ["Ván lướt", "Van luot"],
    "tennis racket": ["Vợt tennis", "Vot tennis"], 
    "bottle": ["Chai", "Chai"], 
    "wine glass": ["Ly rượu", "Ly ruou"],
    "cup": ["Cốc", "Coc"], 
    "fork": ["Nĩa", "Nia"], 
    "knife": ["Dao", "Dao"], 
    "spoon": ["Thìa", "Thia"], 
    "bowl": ["Bát", "Bat"],
    "apple": ["Táo", "Tao"],
    "broccoli": ["Súp lơ", "Sup lo"]
}

# Từ điển màu sắc
color_ranges = {
    "Đỏ": ([0, 100, 100], [10, 255, 255]),
    "Xanh lá": ([35, 100, 100], [85, 255, 255]),
    "Xanh dương": ([100, 100, 100], [130, 255, 255]),
    "Vàng": ([20, 100, 100], [30, 255, 255]),
    "Trắng": ([0, 0, 200], [180, 30, 255]),
    "Đen": ([0, 0, 0], [180, 255, 30]),
}

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phát hiện vật thể - YOLOv8")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Cài đặt theme ttk
        style = ttk.Style()
        style.theme_use('clam')
        self.themes = {'light': {'bg': '#f0f0f0', 'fg': '#333333', 'button_bg': '#e0e0e0'},
                       'dark': {'bg': '#2d2d2d', 'fg': '#ffffff', 'button_bg': '#4a4a4a'}}
        self.current_theme = tk.StringVar(value='light')
        self.root.configure(bg=self.themes['light']['bg'])

        # Font tùy chỉnh
        self.font = font.Font(family="Segoe UI", size=10)
        style.configure("TButton", font=self.font)
        style.configure("TLabel", font=self.font)
        style.configure("TCombobox", font=self.font)

        # Main frame
        self.main_frame = tk.Frame(self.root, bg=self.themes['light']['bg'])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video frame
        self.video_frame = tk.Frame(self.main_frame, bg=self.themes['light']['bg'], relief=tk.RAISED, bd=2)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = tk.Label(self.video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Menu ngữ cảnh
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Lưu ảnh", command=self.save_frame)
        self.context_menu.add_command(label="Tạm dừng/Tiếp tục", command=self.toggle_pause)
        self.video_label.bind("<Button-3>", self.show_context_menu)
        self.video_label.bind("<Button-1>", self.save_object_region)

        # Right panel
        self.right_panel = tk.Frame(self.main_frame, bg=self.themes['light']['bg'])
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Controls frame
        self.controls = ttk.Frame(self.right_panel)
        self.controls.pack(fill=tk.X, pady=5)

        # Theme selector
        ttk.Label(self.controls, text="Giao diện:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.theme_combo = ttk.Combobox(self.controls, textvariable=self.current_theme, values=['light', 'dark'], width=8)
        self.theme_combo.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.theme_combo.bind("<<ComboboxSelected>>", self.change_theme)
        self.add_tooltip(self.theme_combo, "Chọn giao diện sáng hoặc tối")

        # Buttons
        self.start_button = ttk.Button(self.controls, text="Mở Camera", command=self.start_camera)
        self.start_button.grid(row=1, column=0, padx=5, pady=5)
        self.add_tooltip(self.start_button, "Mở camera để phát hiện vật thể (Phím: C)")

        self.stop_button = ttk.Button(self.controls, text="Tắt Camera", command=self.stop_detection)
        self.stop_button.grid(row=1, column=1, padx=5, pady=5)
        self.add_tooltip(self.stop_button, "Dừng camera hoặc video")

        self.pause_button = ttk.Button(self.controls, text="Tạm dừng", command=self.toggle_pause)
        self.pause_button.grid(row=1, column=2, padx=5, pady=5)
        self.add_tooltip(self.pause_button, "Tạm dừng hoặc tiếp tục video/camera (Phím: P)")

        self.video_button = ttk.Button(self.controls, text="Chạy video", command=self.load_video)
        self.video_button.grid(row=2, column=0, padx=5, pady=5)
        self.add_tooltip(self.video_button, "Tải và chạy video")

        self.image_button = ttk.Button(self.controls, text="Chạy ảnh", command=self.load_image)
        self.image_button.grid(row=2, column=1, padx=5, pady=5)
        self.add_tooltip(self.image_button, "Tải và xử lý ảnh")

        self.save_button = ttk.Button(self.controls, text="Lưu ảnh", command=self.save_frame)
        self.save_button.grid(row=2, column=2, padx=5, pady=5)
        self.add_tooltip(self.save_button, "Lưu khung hình hiện tại (Phím: S)")

        self.burst_button = ttk.Button(self.controls, text="Chụp liên tiếp", command=self.burst_capture)
        self.burst_button.grid(row=3, column=0, padx=5, pady=5)
        self.add_tooltip(self.burst_button, "Chụp 5 ảnh liên tiếp")

        self.save_video_button = ttk.Button(self.controls, text="Lưu Video", command=self.start_saving_video)
        self.save_video_button.grid(row=3, column=1, padx=5, pady=5)
        self.add_tooltip(self.save_video_button, "Bắt đầu/dừng lưu video")

        self.save_stats_button = ttk.Button(self.controls, text="Lưu Thống Kê", command=self.save_stats)
        self.save_stats_button.grid(row=3, column=2, padx=5, pady=5)
        self.add_tooltip(self.save_stats_button, "Lưu thống kê vào file CSV")

        # Reset button
        self.reset_button = ttk.Button(self.controls, text="Đặt lại", command=self.reset_app)
        self.reset_button.grid(row=3, column=3, padx=5, pady=5)
        self.add_tooltip(self.reset_button, "Đặt lại chương trình về trạng thái ban đầu (Phím: R)")

        # Configurations
        ttk.Label(self.controls, text="Nhãn:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.label_style = tk.StringVar(value="Có dấu")
        self.label_style_combo = ttk.Combobox(self.controls, textvariable=self.label_style, values=["Có dấu", "Không dấu"], width=10)
        self.label_style_combo.grid(row=4, column=1, padx=5, pady=5)
        self.add_tooltip(self.label_style_combo, "Chọn kiểu nhãn tiếng Việt")

        ttk.Label(self.controls, text="Mô hình:").grid(row=4, column=2, padx=5, pady=5, sticky='w')
        self.model_choice = tk.StringVar(value="yolov8s.pt")
        self.model_combo = ttk.Combobox(self.controls, textvariable=self.model_choice, 
                                        values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"], width=12)
        self.model_combo.grid(row=4, column=3, padx=5, pady=5)
        self.model_combo.bind("<<ComboboxSelected>>", self.load_model)
        self.add_tooltip(self.model_combo, "Chọn mô hình YOLOv8")

        ttk.Label(self.controls, text="Độ tin cậy:").grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.confidence = tk.DoubleVar(value=0.3)
        self.confidence_entry = ttk.Entry(self.controls, textvariable=self.confidence, width=5)
        self.confidence_entry.grid(row=5, column=1, padx=5, pady=5)
        self.add_tooltip(self.confidence_entry, "Đặt ngưỡng độ tin cậy (0.0-1.0)")

        ttk.Label(self.controls, text="Thông báo:").grid(row=5, column=2, padx=5, pady=5, sticky='w')
        self.alert_object = tk.StringVar(value="person")
        self.alert_combo = ttk.Combobox(self.controls, textvariable=self.alert_object, 
                                        values=list(vietnamese_labels.keys()), width=12)
        self.alert_combo.grid(row=5, column=3, padx=5, pady=5)
        self.add_tooltip(self.alert_combo, "Chọn vật thể để phát âm thanh thông báo")

        ttk.Label(self.controls, text="Tỷ lệ khung:").grid(row=6, column=0, padx=5, pady=5, sticky='w')
        self.resize_scale = tk.DoubleVar(value=1.2)
        self.resize_entry = ttk.Entry(self.controls, textvariable=self.resize_scale, width=5)
        self.resize_entry.grid(row=6, column=1, padx=5, pady=5)
        self.add_tooltip(self.resize_entry, "Tỷ lệ thu phóng khung hình (0.1-2.0)")

        ttk.Label(self.controls, text="Ghi chú:").grid(row=6, column=2, padx=5, pady=5, sticky='w')
        self.note = tk.StringVar()
        self.note_entry = ttk.Entry(self.controls, textvariable=self.note, width=15)
        self.note_entry.grid(row=6, column=3, padx=5, pady=5)
        self.add_tooltip(self.note_entry, "Nhập ghi chú cho thống kê")

        # Auto-capture configuration
        ttk.Label(self.controls, text="Chụp tự Midland động:").grid(row=7, column=0, padx=5, pady=5, sticky='w')
        self.auto_capture_object = tk.StringVar(value="person")
        self.auto_capture_combo = ttk.Combobox(self.controls, textvariable=self.auto_capture_object, 
                                              values=list(vietnamese_labels.keys()), width=12)
        self.auto_capture_combo.grid(row=7, column=1, padx=5, pady=5)
        self.add_tooltip(self.auto_capture_combo, "Chọn vật thể để tự động chụp ảnh")

        ttk.Label(self.controls, text="Ngưỡng số lượng:").grid(row=7, column=2, padx=5, pady=5, sticky='w')
        self.auto_capture_threshold = tk.IntVar(value=1)
        self.auto_capture_entry = ttk.Entry(self.controls, textvariable=self.auto_capture_threshold, width=5)
        self.auto_capture_entry.grid(row=7, column=3, padx=5, pady=5)
        self.add_tooltip(self.auto_capture_entry, "Số lượng vật thể tối thiểu để tự động chụp (1-10)")

        # Status and info
        self.status = ttk.Label(self.right_panel, text="Sẵn sàng", foreground="green")
        self.status.pack(pady=5)
        self.text_label = ttk.Label(self.right_panel, text="Văn bản phát hiện: ", wraplength=300)
        self.text_label.pack(pady=5)
        self.fps_label = ttk.Label(self.right_panel, text="FPS: 0.0", foreground="blue")
        self.fps_label.pack(pady=5)

        # Count table
        self.count_frame = ttk.Frame(self.right_panel)
        self.count_frame.pack(fill=tk.BOTH, pady=5)
        self.count_table = ttk.Treeview(self.count_frame, columns=("Loại", "Số lượng"), show="headings", height=5)
        self.count_table.heading("Loại", text="Loại")
        self.count_table.heading("Số lượng", text="Số lượng")
        self.count_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_count = ttk.Scrollbar(self.count_frame, orient=tk.VERTICAL, command=self.count_table.yview)
        self.scrollbar_count.pack(side=tk.RIGHT, fill=tk.Y)
        self.count_table.configure(yscrollcommand=self.scrollbar_count.set)

        # Object details table with color column
        self.details_frame = ttk.Frame(self.right_panel)
        self.details_frame.pack(fill=tk.BOTH, pady=5)
        self.details_table = ttk.Treeview(self.details_frame, 
                                         columns=("Loại", "Số lượng", "Tọa độ", "Độ tin cậy", "Màu sắc"), 
                                         show="headings", height=5)
        self.details_table.heading("Loại", text="Loại")
        self.details_table.heading("Số lượng", text="Số lượng")
        self.details_table.heading("Tọa độ", text="Tọa độ (x1,y1,x2,y2)")
        self.details_table.heading("Độ tin cậy", text="Độ tin cậy")
        self.details_table.heading("Màu sắc", text="Màu sắc")
        self.details_table.column("Màu sắc", width=100)
        self.details_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_details = ttk.Scrollbar(self.details_frame, orient=tk.VERTICAL, command=self.details_table.yview)
        self.scrollbar_details.pack(side=tk.RIGHT, fill=tk.Y)
        self.details_table.configure(yscrollcommand=self.scrollbar_details.set)

        # Chart frame
        self.chart_frame = ttk.Frame(self.right_panel)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Variables
        self.running = False
        self.paused = False
        self.frame = None
        self.cap = None
        self.video_writer = None
        self.is_saving_video = False
        self.object_counts = defaultdict(int)
        self.count_history = defaultdict(list)
        self.time_steps = []
        self.time_step = 0
        self.label_colors = {
            label: (i % 256, (i * 50) % 256, (i * 100) % 256) 
            for i, label in enumerate(vietnamese_labels.keys())
        }
        self.last_frame_time = time.time()
        self.object_details = []
        self.last_detected_objects = set()
        self.last_object_counts = defaultdict(int)
        self.all_detected_objects = set()
        self.bounding_boxes = []
        self.last_auto_capture_time = 0

        # Phím tắt
        self.root.bind('s', lambda event: self.save_frame())
        self.root.bind('c', lambda event: self.start_camera())
        self.root.bind('p', lambda event: self.toggle_pause())
        self.root.bind('r', lambda event: self.reset_app())

        # Tải cấu hình
        self.load_config()

    def reset_app(self):
        """Đặt lại chương trình về trạng thái ban đầu"""
        self.stop_detection()
        self.object_counts.clear()
        self.count_history.clear()
        self.time_steps.clear()
        self.time_step = 0
        self.object_details.clear()
        self.last_detected_objects.clear()
        self.last_object_counts.clear()
        self.all_detected_objects.clear()
        self.bounding_boxes.clear()
        self.last_auto_capture_time = 0
        self.note.set("")
        self.video_label.config(image='')
        self.text_label.config(text="Văn bản phát hiện: ")
        self.fps_label.config(text="FPS: 0.0")
        self.status.config(text="Đã đặt lại chương trình.")
        self.ax.clear()
        self.ax.set_xlabel("Thời gian")
        self.ax.set_ylabel("Số lượng")
        self.ax.set_title("Số lượng vật thể theo thời gian")
        self.canvas.draw()
        self.update_table()
        self.root.after(3000, lambda: self.status.config(text="Sẵn sàng"))

    def toggle_pause(self):
        if self.running:
            self.paused = not self.paused
            self.pause_button.config(text="Tiếp tục" if self.paused else "Tạm dừng")
            self.status.config(text="Đã tạm dừng." if self.paused else "Đang tiếp tục...")

    def add_tooltip(self, widget, text):
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(self.tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()

        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)

    def change_theme(self, event=None):
        theme = self.current_theme.get()
        self.root.configure(bg=self.themes[theme]['bg'])
        self.main_frame.configure(bg=self.themes[theme]['bg'])
        self.video_frame.configure(bg=self.themes[theme]['bg'])
        self.right_panel.configure(bg=self.themes[theme]['bg'])
        self.status.configure(foreground='green' if theme == 'light' else '#00ff00')
        self.fps_label.configure(foreground='blue' if theme == 'light' else '#00b7eb')
        self.text_label.configure(foreground=self.themes[theme]['fg'])
        self.save_config()

    def load_config(self):
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
                self.label_style.set(config.get("label_style", "Có dấu"))
                self.model_choice.set(config.get("model", "yolov8s.pt"))
                self.confidence.set(config.get("confidence", 0.3))
                self.alert_object.set(config.get("alert_object", "person"))
                self.resize_scale.set(config.get("resize_scale", 1.2))
                self.current_theme.set(config.get("theme", "light"))
                self.auto_capture_object.set(config.get("auto_capture_object", "person"))
                self.auto_capture_threshold.set(config.get("auto_capture_threshold", 1))
                self.change_theme()
                self.load_model()
        except FileNotFoundError:
            pass

    def save_config(self):
        config = {
            "label_style": self.label_style.get(),
            "model": self.model_choice.get(),
            "confidence": self.confidence.get(),
            "alert_object": self.alert_object.get(),
            "resize_scale": self.resize_scale.get(),
            "theme": self.current_theme.get(),
            "auto_capture_object": self.auto_capture_object.get(),
            "auto_capture_threshold": self.auto_capture_threshold.get()
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        self.status.config(text="Đã lưu cấu hình.")

    def detect_dominant_color(self, region):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        
        dominant_color = "Không xác định"
        max_count = 0
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)
            count = cv2.countNonZero(mask)
            if count > max_count:
                max_count = count
                dominant_color = color_name
        
        return dominant_color

    def save_object_region(self, event):
        if self.frame is None or not self.bounding_boxes:
            self.status.config(text="Không có vật thể để lưu!")
            return

        x, y = event.x, event.y
        frame_width, frame_height = self.frame.shape[1], self.frame.shape[0]
        label_width, label_height = self.video_label.winfo_width(), self.video_label.winfo_height()
        
        scale_x = frame_width / label_width
        scale_y = frame_height / label_height
        frame_x, frame_y = int(x * scale_x), int(y * scale_y)

        for box in self.bounding_boxes:
            x1, y1, x2, y2, label_vi, conf, suffix, color = box
            if x1 <= frame_x <= x2 and y1 <= frame_y <= y2:
                region = self.frame[y1:y2, x1:x2].copy()
                cv2.rectangle(region, (0, 0), (x2-x1, y2-y1), self.label_colors.get(label_vi, (0, 255, 0)), 2)
                display_label = f"{label_vi}{suffix} ({conf:.2f}) Màu: {color}"
                text_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                cv2.rectangle(region, (0, -text_size[1] - 10), (text_size[0], 0), (255, 255, 255), -1)
                cv2.putText(region, display_label, (0, -5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                filename = f"object_{label_vi}{suffix}_{int(time.time())}.jpg"
                cv2.imwrite(filename, region)
                self.status.config(text=f"Đã lưu vùng vật thể: {filename}")
                self.root.after(3000, lambda: self.status.config(text="Đang phát hiện...") if self.running else None)
                return
        self.status.config(text="Không nhấp vào vùng vật thể nào!")
        self.root.after(3000, lambda: self.status.config(text="Đang phát hiện...") if self.running else None)

    def start_camera(self):
        self.stop_detection()
        self.status.config(text="Đang tải camera...")
        self.root.config(cursor="wait")
        self.cap = cv2.VideoCapture(0)
        self.root.config(cursor="")
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return
        self.running = True
        self.paused = False
        self.detect_loop()

    def stop_detection(self):
        self.running = False
        self.paused = False
        self.stop_saving_video()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status.config(text="Đã dừng phát hiện.")
        self.fps_label.config(text="FPS: 0.0")
        self.save_config()

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            self.stop_detection()
            self.status.config(text="Đang tải video...")
            self.root.config(cursor="wait")
            self.cap = cv2.VideoCapture(file_path)
            self.root.config(cursor="")
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở video!")
                return
            self.running = True
            self.paused = False
            self.detect_loop()
            self.status.config(text=f"Đang chạy video: {file_path}")
            self.save_config()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.status.config(text="Đang tải ảnh...")
            self.root.config(cursor="wait")
            image = cv2.imread(file_path)
            self.root.config(cursor="")
            if image is None:
                messagebox.showerror("Lỗi", "Không thể mở ảnh!")
                return
            self.detect_on_frame(image)
            self.status.config(text=f"Đã xử lý ảnh: {file_path}")

    def save_frame(self, condition_triggered=False):
        if self.frame is not None:
            objects = "_".join([f"{obj}_{count}" for obj, count in self.object_counts.items()])
            prefix = "auto_capture" if condition_triggered else "saved"
            filename = f"{prefix}_{int(time.time())}_{objects}.jpg"
            cv2.imwrite(filename, self.frame)
            self.status.config(text=f"Ảnh đã lưu: {filename}")
            self.root.after(3000, lambda: self.status.config(text="Đang phát hiện...") if self.running else None)

    def burst_capture(self):
        if self.frame is not None:
            for i in range(5):
                objects = "_".join([f"{obj}_{count}" for obj, count in self.object_counts.items()])
                filename = f"burst_{int(time.time())}_{i}_{objects}.jpg"
                cv2.imwrite(filename, self.frame)
                time.sleep(0.2)
            self.status.config(text=f"Đã chụp 5 ảnh liên tiếp.")
        else:
            self.status.config(text="Chưa có khung hình để chụp.")

    def start_saving_video(self):
        if self.frame is not None and not self.is_saving_video:
            filename = f"output_{int(time.time())}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = self.frame.shape[:2]
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            self.is_saving_video = True
            self.save_video_button.config(text="Dừng Lưu Video", command=self.stop_saving_video)
            self.status.config(text=f"Đang lưu video: {filename}")
        elif not self.frame:
            self.status.config(text="Chưa có khung hình để lưu video.")

    def stop_saving_video(self):
        if self.is_saving_video:
            self.is_saving_video = False
            self.video_writer.release()
            self.video_writer = None
            self.save_video_button.config(text="Lưu Video", command=self.start_saving_video)
            self.status.config(text="Đã dừng lưu video.")

    def load_model(self, event=None):
        self.status.config(text="Đang tải mô hình...")
        self.root.config(cursor="wait")
        global model
        try:
            model = YOLO(self.model_choice.get())
            self.status.config(text=f"Đã tải mô hình: {self.model_choice.get()}")
            self.save_config()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mô hình: {str(e)}")
        finally:
            self.root.config(cursor="")

    def save_stats(self):
        if self.count_history:
            df = pd.DataFrame(self.count_history, index=self.time_steps)
            df['Ghi chú'] = self.note.get()
            filename = f"stats_{int(time.time())}.csv"
            df.to_csv(filename)
            self.status.config(text=f"Thống kê đã lưu: {filename}")
        else:
            self.status.config(text="Không có dữ liệu thống kê để lưu.")

    def detect_loop(self):
        if self.running and self.cap:
            if self.paused:
                self.root.after(100, self.detect_loop)
                return
            ret, frame = self.cap.read()
            if not ret:
                self.stop_detection()
                return
            self.detect_on_frame(frame)
            self.root.after(10, self.detect_loop)

    def detect_on_frame(self, frame):
        start_time = time.time()
        scale = self.resize_scale.get()
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        screen_width, screen_height = 1920, 1080
        if frame.shape[1] > screen_width or frame.shape[0] > screen_height:
            scale_factor = min(screen_width / frame.shape[1], screen_height / frame.shape[0])
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        try:
            results = model(frame, conf=self.confidence.get(), iou=0.5)[0]
        except Exception as e:
            self.status.config(text=f"Lỗi phát hiện: {str(e)}")
            return

        self.object_counts.clear()
        self.object_details = []
        self.bounding_boxes = []
        label_instances = defaultdict(int)
        current_objects = set()

        alert_triggered = False
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_en = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            label_list = vietnamese_labels.get(label_en, [label_en, label_en])
            label_vi = label_list[0] if self.label_style.get() == "Có dấu" else label_list[1]
            label_instances[label_vi] += 1
            self.object_counts[label_vi] += 1
            current_objects.add(label_vi)
            self.all_detected_objects.add(label_vi)

            if label_en == self.alert_object.get() and not alert_triggered:
                if winsound is not None:
                    try:
                        winsound.Beep(1000, 500)
                    except Exception:
                        pass
                alert_triggered = True

            suffix = ""
            if label_instances[label_vi] > 1:
                suffix = f" {string.ascii_uppercase[label_instances[label_vi]-1]}"
            display_label = f"{label_vi}{suffix} ({conf:.2f})"

            region = frame[y1:y2, x1:x2]
            dominant_color = self.detect_dominant_color(region) if region.size > 0 else "Không xác định"

            color = self.label_colors.get(label_en, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (255, 255, 255), -1)
            cv2.putText(frame, display_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            self.object_details.append((label_vi, 1, f"({x1},{y1},{x2},{y2})", f"{conf:.2f}", dominant_color))
            self.bounding_boxes.append((x1, y1, x2, y2, label_vi, conf, suffix, dominant_color))

        auto_capture_obj = self.auto_capture_object.get()
        auto_capture_label = vietnamese_labels.get(auto_capture_obj, [auto_capture_obj, auto_capture_obj])[0 if self.label_style.get() == "Có dấu" else 1]
        threshold = self.auto_capture_threshold.get()
        current_time = time.time()
        if auto_capture_label in self.object_counts and self.object_counts[auto_capture_label] >= threshold:
            if current_time - self.last_auto_capture_time >= 2:
                self.save_frame(condition_triggered=True)
                self.last_auto_capture_time = current_time

        for obj, count in self.object_counts.items():
            if self.last_object_counts[obj] != count:
                self.status.config(text=f"Số lượng {obj} thay đổi: {self.last_object_counts[obj]} -> {count}")
                self.root.after(3000, lambda: self.status.config(text="Đang phát hiện...") if self.running else None)
                break
        self.last_object_counts = self.object_counts.copy()

        new_objects = current_objects - self.last_detected_objects
        if new_objects:
            self.status.config(text=f"Phát hiện mới: {', '.join(new_objects)}")
            self.root.after(3000, lambda: self.status.config(text="Đang phát hiện...") if self.running else None)
        self.last_detected_objects = current_objects

        if pytesseract.pytesseract.tesseract_cmd:
            if tessdata_prefix and os.path.exists(os.path.join(tessdata_prefix, 'tessdata')):
                try:
                    text = pytesseract.image_to_string(frame, lang='vie+eng')
                    if text.strip():
                        self.text_label.config(text=f"Văn bản phát hiện: {text.strip()}")
                    else:
                        self.text_label.config(text="Văn bản phát hiện: Không tìm thấy văn bản")
                except pytesseract.TesseractError:
                    self.text_label.config(text="")
                except Exception:
                    self.text_label.config(text="")
            else:
                self.text_label.config(text="")
        else:
            self.text_label.config(text="")

        if self.is_saving_video and self.video_writer is not None:
            self.video_writer.write(frame)

        self.update_table()
        self.frame = frame

        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = img_tk
            self.video_label.config(image=img_tk)
        except Exception as e:
            self.status.config(text=f"Lỗi hiển thị khung hình: {str(e)}")

        current_time = time.time()
        fps = 1 / (current_time - self.last_frame_time) if current_time > self.last_frame_time else 0
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.last_frame_time = current_time

    def update_table(self):
        for i in self.count_table.get_children():
            self.count_table.delete(i)
        for i in self.details_table.get_children():
            self.details_table.delete(i)

        current_counts = defaultdict(int)
        for obj in self.all_detected_objects:
            current_counts[obj] = self.object_counts.get(obj, 0)
            self.count_history[obj].append(current_counts[obj])
            self.count_table.insert('', 'end', values=(obj, current_counts[obj]))

        for detail in self.object_details:
            self.details_table.insert('', 'end', values=detail)

        self.time_step += 1
        self.time_steps.append(self.time_step)
        
        max_history = 100
        for obj in self.count_history:
            if len(self.count_history[obj]) > max_history:
                self.count_history[obj] = self.count_history[obj][-max_history:]
        if len(self.time_steps) > max_history:
            self.time_steps = self.time_steps[-max_history:]

        self.ax.clear()
        colors = ['r', 'b', 'y', 'g', 'm']
        for i, (obj, counts) in enumerate(self.count_history.items()):
            if len(counts) == len(self.time_steps):
                self.ax.plot(self.time_steps, counts, label=obj, color=colors[i % len(colors)])
        self.ax.set_xlabel("Thời gian")
        self.ax.set_ylabel("Số lượng")
        self.ax.set_title("Số lượng vật thể theo thời gian")
        self.ax.legend()
        self.ax.grid(True)
        try:
            self.canvas.draw()
        except Exception as e:
            self.status.config(text=f"Lỗi vẽ biểu đồ: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
    