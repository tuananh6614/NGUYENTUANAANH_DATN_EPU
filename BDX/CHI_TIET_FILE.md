# CHI TIẾT CHỨC NĂNG VÀ NHIỆM VỤ TỪNG FILE

## 📁 FOLDER BDX - Bãi Đỗ Xe (Refactored Version)

---

## 1️⃣ **doan_baidoxe.py** (47 KB)
**📌 File chính của ứng dụng**

### **Chức năng:**
File chính chạy ứng dụng quản lý bãi đỗ xe, sử dụng tất cả modules đã tách.

### **Nhiệm vụ:**
- ✅ Khởi tạo giao diện chính (MainWindow)
- ✅ Quản lý camera vào/ra
- ✅ Nhận dạng biển số xe (ALPR)
- ✅ Xử lý RFID card (xe vào/ra)
- ✅ Tính phí gửi xe
- ✅ Tracking revenue realtime
- ✅ Tracking daily counter (số xe vào trong ngày)
- ✅ Kết nối MQTT với ESP32
- ✅ Xử lý payment confirmation từ terminal
- ✅ Lưu/đọc dữ liệu từ in_records.json

### **Class chính:**
```python
class MainWindow(QMainWindow):
    trigger_shoot_in = Signal(str)      # Signal cho xe vào
    trigger_shoot_out = Signal(str)     # Signal cho xe ra
    trigger_update_revenue = Signal(...) # Signal cập nhật tiền realtime
```

### **Cách chạy:**
```bash
cd BDX
python doan_baidoxe.py
```

---

## 2️⃣ **cau_hinh.py** (2.8 KB)
**📌 Module cấu hình và constants**

### **Chức năng:**
Chứa tất cả cấu hình, constants và dataclass cho app.

### **Nhiệm vụ:**
- ✅ Định nghĩa `UiConfig` dataclass (cấu hình UI)
- ✅ Cung cấp `load_config()`, `save_config()` functions
- ✅ Chứa tất cả constants:
  - `YOLO_MODEL_PATH` - Đường dẫn model YOLO
  - `DIR_IN`, `DIR_OUT` - Thư mục lưu ảnh
  - `YOLO_CONF`, `YOLO_IOU` - Thông số YOLO
  - `FEE_FLAT` - Phí gửi xe
  - `VOTE_FRAMES`, `VOTE_GAP_MS` - Thông số voting
  - `CAP_WIDTH`, `CAP_HEIGHT` - Resolution camera
  - `MAX_WORKERS` - ThreadPool workers
  - `PLATE_RE` - Regex biển số VN

### **Exports:**
```python
from cau_hinh import (
    UiConfig, load_config, save_config,
    YOLO_MODEL_PATH, DIR_IN, DIR_OUT, FEE_FLAT, ...
)
```

### **UiConfig dataclass:**
```python
@dataclass
class UiConfig:
    cam_in_index: int = 0
    cam_out_index: int = -1
    total_slots: int = 50
    mqtt_enable: bool = True
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    gate_id: str = "gate01"
    auto_start_broker: bool = True
    broker_exe: str = "..."
    broker_conf: str = "..."
```

---

## 3️⃣ **cong_cu.py** (6.1 KB)
**📌 Module utility functions**

### **Chức năng:**
Chứa các hàm tiện ích dùng chung trong app.

### **Nhiệm vụ:**
Cung cấp các utility functions cho:

#### **A. Xử lý biển số:**
```python
def plate_similarity(plate1: str, plate2: str) -> float
    # Tính độ tương đồng 2 biển số (0.0 - 1.0)
    # Dùng Levenshtein distance algorithm
```

#### **B. Xử lý ảnh:**
```python
def sharpness_score(gray: np.ndarray) -> float
    # Đánh giá độ sắc nét của ảnh (Laplacian variance)

def enhance_for_plate(bgr: np.ndarray) -> np.ndarray
    # Tăng cường ảnh cho OCR (CLAHE + sharpening)

def np_to_qimage(bgr: np.ndarray) -> QImage
    # Convert numpy array → QImage để hiển thị

def set_pixmap_fit_no_upscale(label: QLabel, img: QImage)
    # Hiển thị ảnh trên QLabel (fit không upscale)
```

#### **C. Camera:**
```python
def list_cameras(max_index=8) -> List[int]
    # Liệt kê tất cả camera có sẵn
```

#### **D. File management:**
```python
def cleanup_old_images(days_old=3)
    # Xóa ảnh cũ hơn N ngày (tự động chạy background)
```

#### **E. Network:**
```python
def is_port_open(host: str, port: int, timeout=0.5) -> bool
    # Kiểm tra port có mở không

def get_local_ips() -> set
    # Lấy tất cả IP local của máy
```

---

## 4️⃣ **camera.py** (6.4 KB)
**📌 Module camera worker thread**

### **Chức năng:**
Quản lý camera trong thread riêng để không block UI.

### **Nhiệm vụ:**
- ✅ Chạy camera trong QThread
- ✅ Capture frames liên tục
- ✅ Tính sharpness score cho từng frame
- ✅ Buffer frames tốt nhất (deque với maxlen=25)
- ✅ Emit frames đã resize cho UI
- ✅ Error handling và auto-recovery
- ✅ Proper cleanup khi stop

### **Class chính:**
```python
class CameraWorker(QThread):
    frame_ready = Signal(QImage)      # Frame để hiển thị UI
    opened = Signal(bool)             # Camera mở thành công?
    error_occurred = Signal(str)      # Lỗi camera

    def run(self):
        # Main camera loop
        # - Capture frame
        # - Calculate sharpness
        # - Buffer best frames
        # - Emit to UI

    def stop(self):
        # Proper shutdown sequence
        # - Set running flag = False
        # - Wait for thread finish
        # - Release camera

    def get_recent_frames(self, n: int, min_score: float, gap_ms: int):
        # Lấy N frames tốt nhất từ buffer
        # Dùng cho ALPR voting

    def best_recent_frame(self, min_score: float):
        # Lấy 1 frame tốt nhất
```

### **Features:**
- Thread-safe với RLock
- Auto-reconnect khi lỗi
- Frame quality scoring
- Memory efficient (chỉ giữ 25 frames)

---

## 5️⃣ **nhan_dien_bien.py** (7.2 KB)
**📌 Module ALPR engine (nhận dạng biển số)**

### **Chức năng:**
Engine nhận dạng biển số xe bằng YOLOv8 + EasyOCR.

### **Nhiệm vụ:**

#### **A. Helper functions:**
```python
def clean_plate_text(raw: str) -> str
    # Làm sạch text OCR (chỉ giữ A-Z, 0-9, -, .)

def order_points(pts)
    # Sắp xếp 4 góc: top-left, top-right, bottom-right, bottom-left

def warp_plate(crop, w=200, h=60)
    # Căn chỉnh biển số thành hình chữ nhật (bird's eye view)
    # Dùng perspective transform
```

#### **B. Class ALPR:**
```python
class ALPR:
    def __init__(self, model_path, conf, iou):
        # - Load YOLO model
        # - Init EasyOCR reader
        # - Init ThreadPoolExecutor
        # - Init cache dict

    def infer_once(self, frame) -> (plate_text, debug_frame):
        # Nhận dạng từ 1 frame:
        # 1. YOLO detect vùng biển
        # 2. Warp + enhance ảnh
        # 3. EasyOCR đọc text
        # 4. Cache kết quả
        # 5. Return plate + debug frame

    def infer_multi(self, frames) -> (plate_text, debug_frame):
        # Nhận dạng từ nhiều frames (voting):
        # 1. Submit tất cả frames vào ThreadPool
        # 2. Mỗi frame chạy infer_once()
        # 3. Voting: biển nào xuất hiện nhiều nhất?
        # 4. Return plate tốt nhất

    def cleanup(self):
        # Shutdown ThreadPoolExecutor
```

### **Features:**
- Multi-threading cho tốc độ
- Cache để tránh OCR lại
- Voting mechanism cho accuracy cao
- Perspective correction
- Image enhancement

---

## 6️⃣ **giao_dien.py** (3.5 KB)
**📌 Module UI helpers và dialogs**

### **Chức năng:**
Cung cấp các components UI và dialog.

### **Nhiệm vụ:**

#### **A. UI Helper functions:**
```python
def qlabel_video_placeholder(text="") -> QLabel
    # Tạo QLabel placeholder cho video
    # - Background dark
    # - Border
    # - Center aligned
    # - Expandable
```

#### **B. Dialogs:**
```python
class SettingsDialog(QDialog):
    # Dialog thiết lập:
    # - Chọn camera IN/OUT
    # - Số slots
    # - MQTT settings (host, port, gate_id)
    # - Auto-start Mosquitto broker
    # - Broker config paths

    def values(self):
        # Return tuple các giá trị từ form
```

---

## 7️⃣ **test_modules.py** (2.8 KB)
**📌 Script test các modules**

### **Chức năng:**
Test tất cả modules để đảm bảo hoạt động đúng.

### **Nhiệm vụ:**
- ✅ Test import từng module
- ✅ Test một số functions cơ bản
- ✅ Hiển thị kết quả [OK] hoặc [FAILED]

### **Cách chạy:**
```bash
cd BDX
python test_modules.py
```

### **Output mong đợi:**
```
============================================================
TESTING REFACTORED MODULES
============================================================

1. Testing cau_hinh.py... [OK]
2. Testing cong_cu.py... [OK]
3. Testing camera.py... [OK]
4. Testing nhan_dien_bien.py... [OK]
5. Testing giao_dien.py... [OK]

============================================================
ALL MODULES TESTED!
============================================================
```

---

## 8️⃣ **run_bdx.py** (3.7 KB)
**📌 Auto-reloader cho development**

### **Chức năng:**
Tự động restart app khi có file Python thay đổi (hot reload).

### **Nhiệm vụ:**
- ✅ Theo dõi thay đổi file Python trong folder
- ✅ Auto-restart app khi detect thay đổi
- ✅ Debounce để tránh restart liên tục
- ✅ Đợi camera release trước khi restart
- ✅ Proper cleanup (terminate → kill nếu cần)

### **Cách dùng:**
```bash
cd BDX
python run_bdx.py
```

### **Features:**
- Watchdog file observer
- Debounce 3 giây
- Camera release wait 2 giây
- Colorama để hiển thị màu sắc
- Ctrl+C để thoát sạch

---

## 9️⃣ **HUONG_DAN_SU_DUNG.md** (6.2 KB)
**📌 Tài liệu hướng dẫn sử dụng**

### **Chức năng:**
Hướng dẫn chi tiết cách sử dụng app và modules.

### **Nội dung:**
- Cấu trúc file sau refactor
- Cách chạy ứng dụng
- Tính năng mới (revenue realtime, daily counter, payment MQTT)
- in_records.json structure
- So sánh 2 phiên bản (cũ vs mới)
- Troubleshooting

---

## 🔄 LUỒNG HOẠT ĐỘNG CHÍNH

### **1. Startup:**
```
main()
  → load_config()
  → MainWindow.__init__()
    → _build_ui()
    → _load_data_from_db()  # Load in_records.json
    → _init_models()        # Init ALPR
    → start_cameras()       # Start camera threads
    → init_mqtt()           # Connect MQTT
    → _start_timers()       # Start UI timers
```

### **2. Xe vào (RFID card scan):**
```
ESP32 → MQTT topic: "parking/gate/{gate_id}/in"
  → _on_message()
  → trigger_shoot_in.emit(card_id)  # Signal
  → _handle_shoot_in(card_id)       # Slot (main thread)
  → on_shoot_in()
    → get_recent_frames()           # Lấy frames tốt nhất
    → alpr.infer_multi(frames)      # ALPR voting
    → Lưu vào _in_records
    → _total_in_count += 1          # Daily counter
    → _save_data_to_db()            # Save to in_records.json
    → Update UI
```

### **3. Xe ra (RFID card scan):**
```
ESP32 → MQTT topic: "parking/gate/{gate_id}/out"
  → _on_message()
  → trigger_shoot_out.emit(card_id)
  → _handle_shoot_out(card_id)
  → on_shoot_out()
    → get_recent_frames()
    → alpr.infer_multi(frames)
    → Tìm record trong _in_records
    → Tính phí (FEE_FLAT)
    → _total_revenue += fee         # Update revenue
    → Remove từ _in_records
    → _save_data_to_db()
    → Update UI
```

### **4. Payment confirmation (từ terminal):**
```
Terminal ESP32 → MQTT topic: "parking/gate/{gate_id}/payment"
  → _on_message()
  → Tìm card trong _in_records
  → _total_revenue += FEE_FLAT
  → trigger_update_revenue.emit()   # Signal
  → _handle_update_revenue()        # Slot (main thread)
  → Update UI REALTIME              # ✅ Không cần restart app
  → _save_data_to_db()
```

---

## 📊 SO SÁNH VỚI FILE CŨ

| Tiêu chí | parking_ui.py (cũ) | BDX (mới) |
|----------|-------------------|-----------|
| **Số files** | 1 file (64 KB) | 9 files (tổng ~77 KB) |
| **File chính** | 64 KB | 47 KB (-26%) |
| **Modules** | Tất cả trong 1 file | Tách thành 5 modules |
| **Maintainability** | Khó (code dài) | Dễ (modular) |
| **Revenue realtime** | ❌ | ✅ |
| **Daily counter** | ❌ | ✅ |
| **Payment MQTT** | ❌ | ✅ |
| **in_records structure** | Flat | Nested (summary + vehicles) |

---

## 🎯 KẾT LUẬN

### **Các file CORE (bắt buộc):**
1. ✅ `doan_baidoxe.py` - File chính
2. ✅ `cau_hinh.py` - Config
3. ✅ `cong_cu.py` - Utils
4. ✅ `camera.py` - Camera worker
5. ✅ `nhan_dien_bien.py` - ALPR
6. ✅ `giao_dien.py` - UI helpers

### **Các file PHỤ (optional):**
7. ⭐ `test_modules.py` - Test script
8. ⭐ `run_bdx.py` - Auto-reloader (dev)
9. ⭐ `HUONG_DAN_SU_DUNG.md` - Documentation

### **Dependencies cần cài:**
```bash
pip install opencv-python ultralytics easyocr pyside6 paho-mqtt watchdog colorama
```

---

**Generated by Claude Code**
**Date: 2025-10-17**
