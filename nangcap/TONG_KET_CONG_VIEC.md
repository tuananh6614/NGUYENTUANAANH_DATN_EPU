# 📋 TỔNG KẾT CÔNG VIỆC - HỆ THỐNG QUẢN LÝ BÃI ĐỖ XE

**Ngày hoàn thành**: 18/10/2025
**Tổng số dòng code**: 1910 dòng (parking_ui.py)

---

## 🎯 TỔNG QUAN HỆ THỐNG

### Kiến trúc tổng thể:
```
┌─────────────────┐         MQTT          ┌──────────────────┐
│   ESP32 #1      │ ◄──────────────────► │                  │
│   (RFID Gate)   │    parking/gate/...   │                  │
│   - Vào/Ra      │                       │   PARKING APP    │
└─────────────────┘                       │   (parking_ui.py)│
                                          │                  │
┌─────────────────┐         MQTT          │   - MQTT Client  │
│   ESP32 #2      │ ◄──────────────────► │   - Camera ALPR  │
│  (Payment Term) │ parking/payment/...   │   - PyQt5 UI     │
│   - Thanh toán  │                       │   - JSON DB      │
└─────────────────┘                       └──────────────────┘
```

---

## ✅ CÁC TÍNH NĂNG ĐÃ HOÀN THÀNH

### 1. **MQTT Integration - Kết nối 2 ESP32**

#### ESP32 #1: Cổng vào/ra (RFID Gate)
**Topics đã implement:**
- `parking/gate/gate01/in` - Xe vào
- `parking/gate/gate01/out` - Xe ra
- `parking/gate/gate01/status` - Trạng thái online/offline
- `parking/gate/gate01/heartbeat` - Kiểm tra kết nối

**Chức năng:**
- ✅ Auto trigger chụp ảnh khi nhận MQTT message
- ✅ Hiển thị trạng thái kết nối real-time
- ✅ Heartbeat monitoring (check mỗi 0.5s)
- ✅ Auto reconnect khi mất kết nối

#### ESP32 #2: Terminal thanh toán (Payment Terminal)
**Topics đã implement:**
- `parking/payment/gate02/card_scanned` - Quẹt thẻ tại terminal
- `parking/payment/gate02/payment_confirmed` - Xác nhận thanh toán
- `parking/payment/gate02/vehicle_info` - Gửi thông tin xe về terminal
- `parking/payment/gate02/status` - Trạng thái
- `parking/payment/gate02/heartbeat` - Kiểm tra kết nối

**Chức năng:**
- ✅ Nhận thẻ từ terminal → Tìm xe trong database
- ✅ Gửi thông tin xe về terminal (biển số, thời gian, phí)
- ✅ Nhận xác nhận thanh toán → Mark `paid=true`
- ✅ Hiển thị màu cam "ĐÃ THANH TOÁN" trên ESP32 khi `paid=true && fee=0`

**File ESP32 tham khảo:**
- `C:\Users\Admin\Documents\PlatformIO\Projects\baidoxe\src\baidoxe.cpp` (Gate)
- `C:\Users\Admin\Documents\PlatformIO\Projects\thanhtoan\src\main.cpp` (Payment)

---

### 2. **Luồng Thanh Toán 3 Bước**

```
BƯỚC 1: XE VÀO
├─ ESP32 Gate → MQTT: parking/gate/gate01/in
├─ App chụp ảnh → ALPR nhận diện biển số
├─ Lưu vào in_records.json:
│  {
│    "card_id": {
│      "plate": "7 3397",
│      "time": "2025-10-18 00:40:55",
│      "paid": false
│    }
│  }
└─ Daily counter +1

BƯỚC 2: THANH TOÁN (TÙY CHỌN)
├─ ESP32 Payment → MQTT: parking/payment/gate02/card_scanned
├─ App tìm thẻ trong in_records
├─ Gửi thông tin về terminal:
│  {
│    "plate": "7 3397",
│    "time_in": "00:40:55",
│    "fee": 3000,  // ✅ fee=0 nếu đã paid
│    "paid": false
│  }
├─ Terminal hiển thị:
│  - Nếu paid=false && fee=3000 → Màn hình XANH "THANH TOÁN"
│  - Nếu paid=true && fee=0 → Màn hình CAM "ĐÃ THANH TOÁN"
├─ Người dùng nhấn nút thanh toán
├─ ESP32 Payment → MQTT: parking/payment/gate02/payment_confirmed
├─ App update:
│  - Mark paid=true
│  - Revenue +3000
│  - GIỮ NGUYÊN trong in_records (xe vẫn trong bãi)
└─ Lưu lại database

BƯỚC 3: XE RA
├─ ESP32 Gate → MQTT: parking/gate/gate01/out
├─ App chụp ảnh → ALPR nhận diện biển số
├─ Tìm thẻ trong in_records
├─ Kiểm tra biển số khớp không:
│  ├─ Nếu similarity < 0.7 → POPUP ĐỎ "BIỂN SỐ KHÔNG KHỚP" → CHẶN XE
│  └─ Nếu OK → Tiếp tục
├─ Kiểm tra đã thanh toán chưa:
│  ├─ Nếu paid=false → POPUP VÀNG "THANH TOÁN" → Thu tiền mặt → Revenue +3000
│  └─ Nếu paid=true → Fee=0 (đã trả rồi)
├─ Xóa khỏi in_records (xe ra khỏi bãi)
├─ Lưu ảnh vào thư mục OUT/
└─ Update UI
```

**Đặc điểm quan trọng:**
- ✅ Revenue chỉ tăng 1 lần/xe (hoặc ở terminal hoặc ở cổng ra)
- ✅ Xe thanh toán ở terminal vẫn ở trong `in_records` cho đến khi ra
- ✅ Popup thanh toán cho phép thu tiền mặt tại cổng ra

---

### 3. **Hệ Thống Cảnh Báo 2 Màu**

#### A. POPUP VÀNG - Thanh toán (parking_ui.py:1323-1375)
**Khi nào xuất hiện:**
- Xe quẹt thẻ RA mà `paid=false` (chưa thanh toán online)

**Thiết kế:**
- 🟡 Màu vàng (#ffcc00) - Không nguy hiểm
- 🟢 Nút xanh lá "OK - Đã thu tiền" (#28a745)
- 💰 Hiển thị: Biển số, thời gian vào, thời gian gửi, phí 3,000 VNĐ
- ⌨️ Nhấn Enter = OK (không cần chuột)

**Hành động:**
- ✅ Nhân viên thu tiền mặt
- ✅ Nhấn OK → Revenue +3,000
- ✅ Xe được phép RA

**Code:**
```python
msg_box.setIcon(QMessageBox.Information)
ok_btn = msg_box.addButton("OK - Đã thu tiền", QMessageBox.AcceptRole)
ok_btn.setDefault(True)  # Enter = OK
```

#### B. POPUP ĐỎ - Biển số sai (parking_ui.py:1291-1343)
**Khi nào xuất hiện:**
- Biển số VÀO ≠ Biển số RA
- Similarity < 0.7 (quá khác biệt)

**Thiết kế:**
- 🔴 Màu đỏ (#ff4444) - Nguy hiểm
- 🔴 Nút đỏ "Hủy" (#dc3545)
- ⛔ Hiển thị: "XE BỊ CHẶN - KHÔNG CHO RA!"
- ⌨️ Nhấn Enter = Hủy

**Hành động:**
- ❌ XE BỊ CHẶN - Không cho ra
- ❌ Không thu tiền
- 🔄 Trả thẻ vào `in_records` (xe vẫn trong bãi)
- 📝 Log cảnh báo

**Code:**
```python
msg_box.setIcon(QMessageBox.Critical)
cancel_btn = msg_box.addButton("Hủy", QMessageBox.RejectRole)
cancel_btn.setDefault(True)  # Enter = Hủy
```

---

### 4. **Database & File Management**

#### Cấu trúc `in_records.json`:
```json
{
  "summary": {
    "total_revenue": 27000,      // Tổng tiền (KHÔNG reset qua ngày)
    "daily_in_count": 12,         // Số xe vào hôm nay (RESET qua ngày)
    "last_date": "2025-10-18"     // Ngày cuối lưu
  },
  "vehicles": {                    // Xe ĐANG trong bãi
    "card_id": {
      "plate": "7 3397",
      "time": "2025-10-18T00:40:55",
      "card_id": "47-1F-14-D8",
      "paid": true/false
    }
  }
}
```

#### Cơ chế lưu/load:
- **Load**: Khi app khởi động (`_load_data_from_db()`)
- **Save**: Mỗi khi có xe vào/ra/thanh toán (`_save_data_to_db()`)
- **Midnight Reset**: Timer check mỗi 60s, reset `daily_in_count` về 0 vào 00:00

#### ⚠️ VẤN ĐỀ BẢO MẬT:
- Chỉ 1 file JSON duy nhất - dễ mất dữ liệu
- ❌ Chưa có backup tự động
- ❌ Chưa có export CSV/Excel
- ❌ Chưa có nút restore

**→ CẦN IMPLEMENT BACKUP SYSTEM** (đang dở dang)

---

### 5. **UI Improvements**

#### A. Đã xóa các nút demo:
- ❌ Nút "Chụp IN" (thay bằng MQTT auto)
- ❌ Nút "Chụp OUT" (thay bằng MQTT auto)
- ❌ Nút "Đồng bộ" (không cần nữa)
- ✅ Chỉ giữ nút "Xóa"

#### B. Hiển thị trạng thái ESP32:
```
┌─ Kết nối MQTT / ESP32 ──────────────┐
│ 🟢 Đã kết nối | ESP32: 2/2 Online  │
│ Broker: 192.168.1.7:1883           │
│ Gate ID: gate01                     │
│                                     │
│ Danh sách ESP32:                    │
│ 🟢 Online | MAC: 80:B5:4E:C6:44:08 │
│    IP: 192.168.1.60 | HB: 2s trước │
│                                     │
│ 🟢 Online | MAC: 34:CD:80:0D:1D:A4 │
│    IP: 192.168.1.88 | HB: 0s trước │
└─────────────────────────────────────┘
```

#### C. Hiển thị tổng doanh thu:
- Thêm field "TỔNG TIỀN" vào UI
- Real-time update khi có thanh toán
- Format: "27,000" (có dấu phẩy ngăn cách)

---

### 6. **Logging System**

#### Đã sửa:
- ❌ Không ghi file `parking_app.log` nữa
- ✅ Chỉ log ra console (`StreamHandler`)
- ✅ Format: `%(asctime)s [%(levelname)s] %(message)s`

#### File đã xóa:
- `parking_app.log` (đã xóa bằng PowerShell)

**Code:**
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # Chỉ console
    ]
)
```

---

## 📁 CẤU TRÚC FILES

```
e:\FIRMWAVE\project\
├── parking_ui.py          (1910 dòng - FILE CHÍNH)
├── config.json            (Cấu hình MQTT, camera, slots)
├── in_records.json        (Database xe trong bãi)
├── LUONG_HOAT_DONG.md    (Documentation luồng hoạt động)
├── TONG_KET_CONG_VIEC.md (File này)
│
├── BDX/                   (Module cũ - đã merge vào parking_ui.py)
│   ├── alpr_module.py
│   ├── camera_module.py
│   └── ...
│
├── IN/                    (Ảnh xe vào)
│   ├── 7_3397_20251018_004055.jpg
│   └── ...
│
├── OUT/                   (Ảnh xe ra)
│   ├── 7_3397_20251018_010843.jpg
│   └── UNREAD/           (Ảnh không đọc được biển số)
│
└── models/                (YOLO models cho ALPR)
    ├── detect.pt
    └── ocr.pt
```

---

## 🔧 DEPENDENCIES

```python
# Python packages (requirements.txt)
PyQt5==5.15.9
opencv-python==4.8.1.78
numpy==1.24.3
ultralytics==8.0.196
paho-mqtt==1.6.1
Pillow==10.0.1
```

**Cài đặt:**
```bash
pip install -r requirements.txt
```

---

## 🚀 CÁCH CHẠY HỆ THỐNG

### 1. Chuẩn bị:
```bash
cd e:\FIRMWAVE\project
```

### 2. Kiểm tra config.json:
```json
{
  "mqtt_host": "192.168.1.7",
  "mqtt_port": 1883,
  "gate_id": "gate01",
  "camera_in_url": "rtsp://...",
  "camera_out_url": "rtsp://...",
  "total_slots": 10
}
```

### 3. Chạy app:
```bash
python parking_ui.py
```

### 4. Kiểm tra kết nối:
- MQTT broker phải chạy (192.168.1.7:1883)
- 2 ESP32 phải online (hiển thị 🟢)
- Camera phải kết nối được

---

## 🐛 CÁC VẤN ĐỀ ĐÃ SỬA

### 1. ❌ ESP32 Payment Terminal không kết nối
**Nguyên nhân:** App không subscribe topics `parking/payment/+/*`

**Đã sửa:**
```python
# Thêm subscription cho payment terminal
client.subscribe("parking/payment/+/heartbeat", qos=0)
client.subscribe("parking/payment/+/status", qos=1)
client.subscribe("parking/payment/+/card_scanned", qos=1)
client.subscribe("parking/payment/+/payment_confirmed", qos=1)
```

### 2. ❌ Popup thanh toán hiển thị 2 lần
**Nguyên nhân:** Hiểu nhầm payment terminal là cổng ra

**Đã sửa:**
- Payment terminal chỉ mark `paid=true`, KHÔNG xóa khỏi `in_records`
- Xe vẫn ở trong bãi cho đến khi quẹt RA tại gate

### 3. ❌ Terminal không hiển thị màu cam "ĐÃ THANH TOÁN"
**Nguyên nhân:** App gửi `fee=3000` ngay cả khi `paid=true`

**Đã sửa:**
```python
vehicle_info = {
    "plate": record["plate"],
    "fee": 0 if already_paid else 3000,  # ✅ Key fix
    "paid": already_paid
}
```

### 4. ❌ Popup cảnh báo không có nút, không phân biệt được
**Đã sửa:**
- Popup VÀNG (thanh toán): Có nút OK, Enter = OK, cho xe ra
- Popup ĐỎ (biển số sai): Có nút Hủy, Enter = Hủy, chặn xe

### 5. ❌ File `parking_app.log` vẫn được tạo
**Đã sửa:**
- Xóa `FileHandler` khỏi logging config
- Xóa file `parking_app.log` bằng PowerShell

---

## 📊 THỐNG KÊ CODE

```
Tổng số dòng: 1910 lines
Tổng số functions: ~50 functions
Tổng số classes: 3 classes (ALPRModule, CamThread, MainWindow)

Phần chính:
- ALPR Module: ~200 dòng
- Camera Module: ~100 dòng
- MQTT Handlers: ~300 dòng
- UI Setup: ~200 dòng
- Event Handlers: ~400 dòng
- Database: ~100 dòng
- Utils: ~100 dòng
```

---

## ⚠️ CÔNG VIỆC ĐANG DỞ DANG

### 1. **Backup System** (Chưa làm)
- [ ] Auto backup mỗi khi save
- [ ] Daily backup vào 00:00
- [ ] Export CSV cho báo cáo
- [ ] Nút Backup/Restore trong UI

### 2. **Reports/Analytics** (Chưa làm)
- [ ] Báo cáo doanh thu theo ngày/tuần/tháng
- [ ] Thống kê số xe vào/ra
- [ ] Export Excel
- [ ] Biểu đồ

### 3. **Error Handling** (Còn thiếu sót)
- [ ] Xử lý khi mất kết nối MQTT lâu
- [ ] Xử lý khi camera die
- [ ] Auto restart khi crash

### 4. **Testing** (Chưa có)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Load testing

---

## 📝 GHI CHÚ QUAN TRỌNG

### 1. **Database chỉ có 1 file JSON - DỄ MẤT DỮ LIỆU!**
- Nếu xóa `in_records.json` → Mất hết dữ liệu
- Nếu file corrupt → App không chạy được
- **→ CẦN BACKUP NGAY!**

### 2. **Revenue KHÔNG BAO GIỜ RESET**
- `total_revenue` tích lũy mãi
- Chỉ có `daily_in_count` reset vào 00:00
- Nếu muốn reset revenue → Phải sửa thủ công trong file JSON

### 3. **Biển số similarity threshold = 0.7**
- Nếu similarity < 0.7 → Cảnh báo đỏ và chặn xe
- Có thể điều chỉnh trong code nếu cần:
```python
if sim < 0.7:  # Thay đổi giá trị này nếu cần
```

### 4. **Enter = OK/Hủy**
- Popup vàng: Enter = OK (cho xe ra)
- Popup đỏ: Enter = Hủy (chặn xe)
- Không cần dùng chuột

---

## 🎓 KIẾN THỨC CẦN NHỚ

### Luồng dữ liệu MQTT:
```
ESP32 → MQTT Broker → Python App → JSON File
  ↓         ↓              ↓           ↓
Quẹt    Publish      Subscribe    Save
thẻ     message      message      data
```

### Cấu trúc in_records.json:
```
summary: {revenue, count, date}  → Thống kê tổng
vehicles: {card_id: {...}}       → Xe đang trong bãi
```

### Khi qua ngày mới (00:00):
```
daily_in_count: 12 → 0  (RESET)
total_revenue: 27000 → 27000  (GIỮ NGUYÊN)
vehicles: {...} → {...}  (GIỮ NGUYÊN)
```

---

## 📞 HỖ TRỢ & TÀI LIỆU

### Files tài liệu:
1. `LUONG_HOAT_DONG.md` - Mô tả chi tiết luồng hoạt động
2. `TONG_KET_CONG_VIEC.md` - File này
3. Code comments trong `parking_ui.py`

### ESP32 source code:
1. `C:\Users\Admin\Documents\PlatformIO\Projects\baidoxe\src\baidoxe.cpp`
2. `C:\Users\Admin\Documents\PlatformIO\Projects\thanhtoan\src\main.cpp`

### MQTT Topics Reference:
```
# Gate ESP32
parking/gate/gate01/in
parking/gate/gate01/out
parking/gate/gate01/status
parking/gate/gate01/heartbeat

# Payment Terminal ESP32
parking/payment/gate02/card_scanned
parking/payment/gate02/payment_confirmed
parking/payment/gate02/vehicle_info
parking/payment/gate02/status
parking/payment/gate02/heartbeat
```

---

## ✅ CHECKLIST TỰ BẢO TRÌ

### Hàng ngày:
- [ ] Kiểm tra 2 ESP32 online (🟢)
- [ ] Backup file `in_records.json` (thủ công)
- [ ] Kiểm tra camera hoạt động
- [ ] Xem log có lỗi không

### Hàng tuần:
- [ ] Export dữ liệu ra Excel (thủ công)
- [ ] Xóa ảnh cũ trong IN/OUT/ (nếu đầy)
- [ ] Restart app để clear memory

### Hàng tháng:
- [ ] Backup toàn bộ thư mục project
- [ ] Kiểm tra dung lượng ổ đĩa
- [ ] Update models ALPR (nếu có mới)

---

**Tài liệu được tạo bởi: Claude (Anthropic)**
**Ngày: 18/10/2025**
**Version: 1.0**
