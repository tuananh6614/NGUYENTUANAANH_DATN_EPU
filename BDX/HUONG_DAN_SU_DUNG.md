# HƯỚNG DẪN SỬ DỤNG - PHẦN MỀM QUẢN LÝ BÃI ĐỖ XE

## 📁 CẤU TRÚC FILE SAU KHI REFACTOR

### **File chính:**
- **`doan_baidoxe.py`** (47 KB) - File chính mới, sử dụng các module đã tách
  - ✅ Code ngắn gọn hơn nhiều
  - ✅ Có đầy đủ chức năng revenue realtime update
  - ✅ Có in_records.json với structure mới
  - ✅ Import từ các module đã tách

- **`parking_ui.py`** (64 KB) - File cũ, giữ nguyên làm backup
  - ⚠️ Chưa có revenue realtime fix
  - ⚠️ Code dài và phức tạp

### **Các module đã tách:**
1. **`cau_hinh.py`** (2.8 KB) - Cấu hình và constants
   - UiConfig dataclass
   - load_config(), save_config()
   - Tất cả constants (YOLO_MODEL_PATH, DIR_IN, DIR_OUT, FEE_FLAT, etc.)

2. **`cong_cu.py`** (6.1 KB) - Utility functions
   - plate_similarity() - Tính độ tương đồng biển số
   - sharpness_score() - Đánh giá độ sắc nét
   - enhance_for_plate() - Tăng cường ảnh cho OCR
   - list_cameras() - Liệt kê camera
   - cleanup_old_images() - Xóa ảnh cũ
   - get_local_ips(), is_port_open() - Network utils

3. **`camera.py`** (6.4 KB) - CameraWorker class
   - Class CameraWorker(QThread) đầy đủ
   - Camera frame buffering + sharpness scoring
   - Thread-safe stop/cleanup

4. **`nhan_dien_bien.py`** (7.2 KB) - ALPR engine
   - clean_plate_text() - Làm sạch text biển số
   - order_points(), warp_plate() - Perspective transform
   - Class ALPR với YOLOv8 + EasyOCR
   - Cache management + multi-threading

5. **`giao_dien.py`** (3.5 KB) - UI helpers + dialogs
   - qlabel_video_placeholder() - Tạo QLabel cho video
   - Class SettingsDialog - Dialog thiết lập

### **File test và tiện ích:**
- **`test_modules.py`** (2.8 KB) - Test tất cả modules
- **`run.py`** (3.7 KB) - Script chạy app cũ

---

## 🚀 CÁCH CHẠY ỨNG DỤNG

### **Option 1: Chạy file mới (RECOMMENDED)**
```bash
python doan_baidoxe.py
```

### **Option 2: Chạy file cũ (backup)**
```bash
python parking_ui.py
```

---

## ✨ TÍNH NĂNG MỚI TRONG `doan_baidoxe.py`

### **1. Revenue Tracking Realtime**
- ✅ Tổng tiền thu được hiển thị realtime trên UI
- ✅ Không cần tắt/bật lại app để thấy cập nhật
- ✅ Sử dụng Qt Signal/Slot pattern để thread-safe

### **2. in_records.json với Structure Mới**
**Cấu trúc mới:**
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
- ✅ Dễ quản lý revenue và counter
- ✅ Dễ reset counter hàng ngày
- ✅ Backward compatible với format cũ

### **3. Code Ngắn Gọn Hơn**
- **parking_ui.py**: 64 KB (1896 dòng) - Tất cả code trong 1 file
- **doan_baidoxe.py**: 47 KB (~1350 dòng) - Import từ modules
- **Giảm**: ~25% code nhờ tái sử dụng modules

### **4. Dễ Bảo Trì Hơn**
- Mỗi module có 1 nhiệm vụ rõ ràng
- Dễ debug - biết lỗi ở module nào
- Dễ mở rộng - thêm tính năng vào module tương ứng

---

## 🔧 MQTT PAYMENT CONFIRMATION

`doan_baidoxe.py` hỗ trợ **payment confirmation từ terminal**:

**Topic:** `parking/gate/{gate_id}/payment`

**Payload:**
```json
{
  "card_id": "33-00-61-F5",
  "mac": "AA:BB:CC:DD:EE:FF"
}
```

**Khi nhận được payment:**
1. Tìm thẻ trong in_records
2. Cập nhật revenue (+3000 VND)
3. Mark thẻ là đã thanh toán
4. Lưu vào in_records.json
5. **Cập nhật UI realtime** (không cần tắt/bật app)

---

## 📊 SO SÁNH 2 PHIÊN BẢN

| Tính năng | parking_ui.py (cũ) | doan_baidoxe.py (mới) |
|-----------|--------------------|-----------------------|
| Kích thước file | 64 KB | 47 KB |
| Số dòng code | ~1896 | ~1350 |
| Revenue realtime | ❌ Không | ✅ Có |
| in_records structure | ❌ Flat | ✅ Nested (summary + vehicles) |
| Daily counter tracking | ❌ Không | ✅ Có |
| Modular code | ❌ Không | ✅ Có (5 modules) |
| Dễ bảo trì | ⚠️ Khó | ✅ Dễ |
| Payment confirmation | ❌ Không | ✅ Có |

---

## 🧪 TEST CÁC MODULE

Chạy file test để kiểm tra các modules:
```bash
python test_modules.py
```

**Kết quả mong đợi:**
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

## 📝 LƯU Ý

1. **File backup:**
   - `parking_ui.py.backup` - Backup nguyên bản
   - `parking_ui.py` - File cũ (giữ làm reference)

2. **Migration từ file cũ:**
   - Nếu bạn đang dùng `parking_ui.py`, chỉ cần chạy `doan_baidoxe.py`
   - File mới tự động đọc được in_records.json format cũ
   - Sau đó sẽ tự động chuyển sang format mới khi save

3. **Nếu có lỗi:**
   - Kiểm tra các module đã cài đủ chưa (opencv, ultralytics, easyocr, pyside6, paho-mqtt)
   - Chạy `python test_modules.py` để test từng module
   - Check log file để xem lỗi chi tiết

---

## 🎯 KHUYẾN NGHỊ

**Nên dùng:** `doan_baidoxe.py` (file mới)

**Lý do:**
1. ✅ Code ngắn gọn, dễ đọc
2. ✅ Có revenue tracking realtime
3. ✅ Có daily counter tracking
4. ✅ Dễ mở rộng thêm tính năng
5. ✅ Dễ debug khi có lỗi

---

## 📞 HỖ TRỢ

Nếu có vấn đề, kiểm tra:
1. Log khi chạy app
2. File `in_records.json` structure
3. Test từng module bằng `test_modules.py`

---

**Generated by Claude Code - Phiên bản refactored**
**Date: 2025-10-17**
