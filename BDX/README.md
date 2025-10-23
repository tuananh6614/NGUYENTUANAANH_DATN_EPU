# 🚗 BDX - BÃI ĐỖ XE (Parking Management System)

Phần mềm quản lý bãi đỗ xe với nhận dạng biển số tự động (ALPR) và kết nối ESP32 qua MQTT.

## 📁 CẤU TRÚC FOLDER

```
BDX/
├── doan_baidoxe.py          # ⭐ File chính - Chạy app
├── cau_hinh.py              # 📝 Config + constants
├── cong_cu.py               # 🛠️ Utility functions
├── camera.py                # 📷 Camera worker thread
├── nhan_dien_bien.py        # 🔍 ALPR engine (YOLOv8 + EasyOCR)
├── giao_dien.py             # 🎨 UI helpers + dialogs
├── test_modules.py          # 🧪 Test script
├── run_bdx.py               # 🔄 Auto-reloader (development)
├── README.md                # 📖 File này
├── HUONG_DAN_SU_DUNG.md     # 📚 Hướng dẫn chi tiết
└── CHI_TIET_FILE.md         # 📋 Chi tiết từng file
```

---

## 🚀 QUICK START

### **1. Cài đặt dependencies:**
```bash
pip install opencv-python ultralytics easyocr pyside6 paho-mqtt numpy
```

### **2. Chạy app:**
```bash
cd BDX
python doan_baidoxe.py
```

### **3. Development mode (auto-reload):**
```bash
cd BDX
python run_bdx.py
```

---

## ✨ TÍNH NĂNG CHÍNH

### **1. Nhận dạng biển số (ALPR)**
- ✅ YOLOv8 để detect vùng biển
- ✅ EasyOCR để đọc ký tự
- ✅ Multi-frame voting cho accuracy cao
- ✅ Perspective correction
- ✅ Image enhancement

### **2. Quản lý xe vào/ra**
- ✅ Thẻ RFID để quản lý xe
- ✅ Camera 2 ngõ (vào/ra)
- ✅ Tự động chụp ảnh khi có xe
- ✅ Tính phí gửi xe
- ✅ Tracking slot còn trống

### **3. Revenue Tracking Realtime** ⭐ NEW
- ✅ Tổng tiền thu được hiển thị realtime
- ✅ Không cần restart app để thấy cập nhật
- ✅ Sử dụng Qt Signal/Slot pattern (thread-safe)

### **4. Daily Counter Tracking** ⭐ NEW
- ✅ Số xe vào trong ngày
- ✅ Tự động reset vào 00:00
- ✅ Lưu trong in_records.json

### **5. MQTT Integration**
- ✅ Kết nối với ESP32
- ✅ Nhận lệnh xe vào/ra
- ✅ Nhận payment confirmation từ terminal
- ✅ Heartbeat monitoring
- ✅ Multi-device support

---

## 📊 in_records.json STRUCTURE

**Format mới:**
```json
{
  "summary": {
    "total_revenue": 24000,
    "daily_in_count": 4,
    "last_date": "2025-10-17"
  },
  "vehicles": {
    "33-00-61-F5": {
      "plate": "365-D 7769",
      "time": "2025-10-17T15:03:10",
      "card_id": "33-00-61-F5",
      "paid": true
    }
  }
}
```

**Lợi ích:**
- Dễ tracking revenue và counter
- Dễ reset counter hàng ngày
- Backward compatible với format cũ

---

## 🔌 MQTT TOPICS

**Base topic:** `parking/gate/{gate_id}/`

### **Subscribe (app nhận):**
1. `in` - Xe vào
   ```json
   {"card_id": "33-00-61-F5", "mac": "AA:BB:CC:DD:EE:FF"}
   ```

2. `out` - Xe ra
   ```json
   {"card_id": "33-00-61-F5", "mac": "AA:BB:CC:DD:EE:FF"}
   ```

3. `payment` - Xác nhận thanh toán ⭐ NEW
   ```json
   {"card_id": "33-00-61-F5", "mac": "AA:BB:CC:DD:EE:FF"}
   ```

4. `heartbeat` - Heartbeat từ ESP32
   ```json
   {"mac": "AA:BB:CC:DD:EE:FF", "ip": "192.168.1.100"}
   ```

5. `status` - Trạng thái ESP32
   ```json
   {"mac": "AA:BB:CC:DD:EE:FF", "online": true, "ip": "192.168.1.100"}
   ```

### **Publish (app gửi):**
- `event` - Sự kiện từ app
- `stats` - Thống kê

---

## 🎯 LUỒNG HOẠT ĐỘNG

### **Xe vào:**
```
1. Quẹt thẻ RFID → ESP32 gửi MQTT "in"
2. App nhận signal → Chụp ảnh camera
3. ALPR nhận dạng biển số
4. Lưu vào in_records
5. Daily counter +1
6. Save to JSON
7. Update UI
```

### **Xe ra:**
```
1. Quẹt thẻ RFID → ESP32 gửi MQTT "out"
2. App nhận signal → Chụp ảnh camera
3. ALPR nhận dạng biển số
4. Tìm record trong in_records
5. Tính phí (3000 VND)
6. Revenue += 3000
7. Remove khỏi in_records
8. Save to JSON
9. Update UI
```

### **Payment từ terminal:** ⭐ NEW
```
1. Thanh toán → Terminal gửi MQTT "payment"
2. App nhận signal
3. Tìm thẻ trong in_records
4. Revenue += 3000
5. Mark thẻ là đã thanh toán
6. Save to JSON
7. ✨ Update UI REALTIME (không cần restart)
```

---

## 📚 TÀI LIỆU CHI TIẾT

1. **[HUONG_DAN_SU_DUNG.md](HUONG_DAN_SU_DUNG.md)** - Hướng dẫn sử dụng đầy đủ
2. **[CHI_TIET_FILE.md](CHI_TIET_FILE.md)** - Chi tiết chức năng từng file

---

## 🔧 CẤU HÌNH

File cấu hình: `config.json` (tự động tạo nếu chưa có)

```json
{
  "cam_in_index": 0,
  "cam_out_index": 1,
  "total_slots": 50,
  "mqtt_enable": true,
  "mqtt_host": "127.0.0.1",
  "mqtt_port": 1883,
  "gate_id": "gate01",
  "auto_start_broker": true,
  "broker_exe": "C:\\Program Files\\mosquitto\\mosquitto.exe",
  "broker_conf": "E:\\FIRMWAVE\\project\\mosquitto.conf"
}
```

---

## 🧪 TEST

Test tất cả modules:
```bash
cd BDX
python test_modules.py
```

**Output mong đợi:**
```
1. Testing cau_hinh.py... [OK]
2. Testing cong_cu.py... [OK]
3. Testing camera.py... [OK]
4. Testing nhan_dien_bien.py... [OK]
5. Testing giao_dien.py... [OK]
```

---

## 🐛 TROUBLESHOOTING

### **Lỗi: "Không tìm thấy YOLO model"**
→ Check đường dẫn trong `cau_hinh.py`:
```python
YOLO_MODEL_PATH = r"E:\FIRMWAVE\...\license_plate_detector.pt"
```

### **Lỗi: "Camera không mở được"**
→ Kiểm tra:
1. Camera có kết nối không?
2. Camera index đúng không? (0, 1, 2...)
3. App khác đang dùng camera không?

### **Lỗi: "MQTT không connect"**
→ Kiểm tra:
1. Mosquitto broker đã chạy chưa?
2. Host/port đúng chưa?
3. Firewall có block không?

### **Revenue không update realtime**
→ ✅ FIXED! Version mới đã fix bằng Signal/Slot pattern

### **Daily counter bị reset giữa ngày**
→ Kiểm tra `in_records.json`:
- `last_date` phải đúng ngày hiện tại
- Nếu sai, file bị corrupt → xóa và chạy lại

---

## 📊 SO SÁNH VỚI PHIÊN BẢN CŨ

| Feature | parking_ui.py (cũ) | BDX (mới) |
|---------|-------------------|-----------|
| File size | 64 KB | 47 KB main + 5 modules |
| Revenue realtime | ❌ | ✅ |
| Daily counter | ❌ | ✅ |
| Payment MQTT | ❌ | ✅ |
| Modular code | ❌ | ✅ |
| Easy maintenance | ⚠️ | ✅ |
| in_records structure | Flat | Nested |

---

## 🎓 KIẾN TRÚC

```
┌─────────────────────────────────────────────┐
│         doan_baidoxe.py (MainWindow)        │
│  ┌───────────────────────────────────────┐  │
│  │  UI Layer (PySide6)                   │  │
│  └───────────────────────────────────────┘  │
│               ↓         ↓         ↓          │
│  ┌─────────┐ ┌────────┐ ┌────────────────┐  │
│  │ Camera  │ │  ALPR  │ │  MQTT Client   │  │
│  │ Threads │ │ Engine │ │  (paho-mqtt)   │  │
│  └─────────┘ └────────┘ └────────────────┘  │
└─────────────────────────────────────────────┘
       ↓              ↓              ↓
┌──────────┐   ┌───────────┐  ┌──────────┐
│ cv2      │   │ YOLOv8    │  │ ESP32    │
│ VideoCapture│ │ EasyOCR   │  │ Devices  │
└──────────┘   └───────────┘  └──────────┘
```

---

## 🔐 BẢO MẬT

⚠️ **Lưu ý:**
- File này dành cho mục đích học tập và development
- Chưa có authentication cho MQTT
- Chưa encrypt dữ liệu
- Production cần thêm security layers

---

## 📝 CHANGELOG

### **v2.0 (2025-10-17) - BDX Refactored**
- ✨ Tách code thành 5 modules
- ✨ Revenue tracking realtime
- ✨ Daily counter tracking
- ✨ Payment confirmation MQTT
- ✨ in_records.json structure mới
- 🐛 Fixed revenue update bug
- 📚 Tài liệu đầy đủ

### **v1.0 - parking_ui.py**
- ✅ Basic ALPR
- ✅ MQTT integration
- ✅ Camera management
- ⚠️ Monolithic code (64 KB)
- ❌ No revenue tracking
- ❌ No daily counter

---

## 🤝 ĐÓNG GÓP

File này được tạo bởi Claude Code để refactor parking_ui.py.

**Mục tiêu:**
- ✅ Code ngắn gọn, dễ đọc
- ✅ Modular, dễ bảo trì
- ✅ Thêm tính năng mới
- ✅ Không làm mất chức năng cũ

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:
1. Đọc [HUONG_DAN_SU_DUNG.md](HUONG_DAN_SU_DUNG.md)
2. Đọc [CHI_TIET_FILE.md](CHI_TIET_FILE.md)
3. Chạy `python test_modules.py` để test
4. Check log output khi chạy app

---

**🎯 Sẵn sàng sử dụng!**

```bash
cd BDX
python doan_baidoxe.py
```

---

**Generated by Claude Code**
**Date: 2025-10-17**
