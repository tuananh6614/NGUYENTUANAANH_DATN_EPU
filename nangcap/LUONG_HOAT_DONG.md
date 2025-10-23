# 🚗 LUỒNG HOẠT ĐỘNG HỆ THỐNG BÃI ĐỖ XE

## 📋 Tổng quan hệ thống

### **Thiết bị:**
1. **ESP32 BAIDOXE (gate01)** - Cổng vào/ra bãi
   - 2 RFID: RFID-A (IN) và RFID-B (OUT)
   - Gửi MQTT topics: `parking/gate/gate01/in`, `parking/gate/gate01/out`

2. **ESP32 THANHTOAN (gate02)** - Máy thanh toán
   - 1 RFID + TFT màn hình
   - Gửi MQTT topics: `parking/payment/gate02/card_scanned`, `parking/payment/gate02/payment_confirmed`

3. **Python App** - Server xử lý ALPR và quản lý
   - Nhận dạng biển số (YOLOv8 + EasyOCR)
   - Quản lý in_records.json
   - Tracking revenue realtime

---

## 🎯 LUỒNG HOẠT ĐỘNG CHI TIẾT

### **1. XE VÀO BÃI (IN)** 🚙➡️

```
┌─────────────┐
│ 1. Quẹt thẻ │ Khách quẹt thẻ RFID tại RFID-A (cổng vào)
│   RFID-A    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 2. ESP32 gửi MQTT                   │
│    Topic: parking/gate/gate01/in    │
│    Payload: {"card_id": "33-00-61-F5"}
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 3. Python App nhận signal           │
│    - Trigger camera IN              │
│    - Chụp ảnh xe                    │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 4. ALPR nhận dạng biển số           │
│    - YOLOv8 detect vùng biển        │
│    - EasyOCR đọc ký tự              │
│    - Multi-frame voting             │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 5. Lưu vào in_records               │
│    {                                │
│      "33-00-61-F5": {               │
│        "plate": "365-D 7769",       │
│        "time": "2025-10-17T15:03:10"│
│        "card_id": "33-00-61-F5",    │
│        "paid": false                │
│      }                              │
│    }                                │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 6. Update counters & UI             │
│    - Daily counter +1               │
│    - Used slots +1                  │
│    - Save to in_records.json        │
│    - Update UI                      │
└─────────────────────────────────────┘
```

**Kết quả:**
- ✅ Xe được ghi nhận vào bãi
- ✅ Thẻ RFID được liên kết với biển số
- ✅ Slot count được cập nhật

---

### **2. THANH TOÁN TẠI TERMINAL** 💳

```
┌─────────────┐
│ 1. Quẹt thẻ │ Khách quẹt thẻ tại payment terminal
│  Terminal   │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 2. ESP32 gửi MQTT                        │
│    Topic: parking/payment/gate02/card_scanned
│    Payload: {"card_id": "33-00-61-F5"}   │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 3. Python App tìm thẻ trong in_records   │
│    - Check card_id có trong records không│
│    - Lấy thông tin: plate, time_in, fee  │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 4. App gửi thông tin xe về Terminal      │
│    Topic: parking/payment/gate02/vehicle_info
│    Payload (chưa thanh toán): {          │
│      "plate": "365-D 7769",              │
│      "time_in": "15:03:10",              │
│      "time_out": "16:30:00",             │
│      "fee": 3000,                        │
│      "paid": false                       │
│    }                                     │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 5. Terminal hiển thị thông tin           │
│    - Biển số: 365-D 7769                 │
│    - Số tiền: 3000 VND                   │
│    - Countdown 10 giây                   │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 6. Auto thanh toán sau 10s               │
│    - Hiển thị animation success          │
│    - Gửi xác nhận về app                 │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 7. ESP32 gửi payment_confirmed           │
│    Topic: parking/payment/gate02/payment_confirmed
│    Payload: {"card_id": "33-00-61-F5"}   │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 8. Python App xử lý thanh toán           │
│    - Mark record: paid = true            │
│    - Revenue += 3000                     │
│    - GIỮ trong in_records (chưa xóa!)    │
│    - Update UI realtime                  │
│    - Save to in_records.json             │
└──────────────────────────────────────────┘
```

**Kết quả:**
- ✅ Revenue được cập nhật ngay
- ✅ Thẻ được đánh dấu đã thanh toán (`paid = true`)
- ⚠️ Xe VẪN ở trong bãi (chưa ra)
- ⚠️ Record VẪN trong in_records (để xe ra sau)

---

### **2B. QUẸT LẠI THẺ ĐÃ THANH TOÁN** 🟠

```
┌─────────────┐
│ 1. Quẹt lại │ Khách quẹt lại thẻ đã thanh toán
│  Terminal   │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 2. ESP32 gửi card_scanned                │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 3. App check: paid = true                │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 4. App gửi thông tin ĐÃ THANH TOÁN       │
│    Payload: {                            │
│      "plate": "365-D 7769",              │
│      "fee": 0,          ⬅️ Đã thanh toán │
│      "paid": true       ⬅️ Quan trọng!   │
│    }                                     │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ 5. Terminal hiển thị MÀN HÌNH CAM 🟠     │
│    - Icon checkmark trong vòng cam       │
│    - Text: "ĐÃ THANH TOÁN RỒI!"         │
│    - Biển số: 365-D 7769                 │
│    - Số tiền: 0 VND                      │
│    - Tự động đóng sau 3 giây             │
└──────────────────────────────────────────┘
```

**Kết quả:**
- ✅ Không cho thanh toán lại
- ✅ Hiển thị màn hình cam
- ✅ Revenue không tăng thêm

---

### **3. XE RA KHỎI BÃI (OUT)** 🚙⬅️

#### **3A. Trường hợp ĐÃ THANH TOÁN tại Terminal**

```
┌─────────────┐
│ 1. Quẹt thẻ │ Khách quẹt thẻ tại RFID-B (cổng ra)
│   RFID-B    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 2. ESP32 gửi MQTT                   │
│    Topic: parking/gate/gate01/out   │
│    Payload: {"card_id": "33-00-61-F5"}
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 3. Python App nhận signal           │
│    - Trigger camera OUT             │
│    - Chụp ảnh xe                    │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 4. ALPR nhận dạng biển số           │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 5. Tìm record trong in_records      │
│    - Check: paid = true             │
│    - Xe đã thanh toán rồi!          │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 6. Xử lý xe ra                      │
│    - Tính thời gian gửi xe          │
│    - Fee = 0 (đã trả rồi)           │
│    - KHÔNG cộng revenue             │
│    - XÓA khỏi in_records            │
│    - Used slots -1                  │
│    - Update UI: "Đã thanh toán"     │
└─────────────────────────────────────┘
```

**Kết quả:**
- ✅ Xe được phép ra
- ✅ Không thu phí (đã trả tại terminal)
- ✅ Slot được giải phóng
- ✅ Revenue không đổi (đã tính lúc thanh toán)

---

#### **3B. Trường hợp CHƯA THANH TOÁN**

```
┌─────────────┐
│ 1. Quẹt thẻ │ Khách quẹt thẻ tại RFID-B (cổng ra)
│   RFID-B    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 2-4. Giống flow đã thanh toán       │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 5. Tìm record trong in_records      │
│    - Check: paid = false            │
│    - Xe CHƯA thanh toán!            │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ 6. Xử lý thanh toán tại cổng        │
│    - Tính thời gian gửi xe          │
│    - Fee = 3000 VND                 │
│    - Revenue += 3000                │
│    - XÓA khỏi in_records            │
│    - Used slots -1                  │
│    - Update UI                      │
└─────────────────────────────────────┘
```

**Kết quả:**
- ✅ Xe được phép ra
- ✅ Thu phí tại cổng
- ✅ Slot được giải phóng
- ✅ Revenue được cập nhật

---

## 📊 in_records.json Structure

```json
{
  "summary": {
    "total_revenue": 24000,
    "daily_in_count": 8,
    "last_date": "2025-10-18"
  },
  "vehicles": {
    "33-00-61-F5": {
      "plate": "365-D 7769",
      "time": "2025-10-18T15:03:10",
      "card_id": "33-00-61-F5",
      "paid": true
    },
    "AA-BB-CC-DD": {
      "plate": "29-A1 12345",
      "time": "2025-10-18T16:20:00",
      "card_id": "AA-BB-CC-DD",
      "paid": false
    }
  }
}
```

**Field `paid`:**
- `false` - Chưa thanh toán → Thu phí khi ra
- `true` - Đã thanh toán tại terminal → Không thu phí khi ra

---

## 🔄 MQTT Topics Summary

### **ESP32 BAIDOXE → App**
| Topic | Payload | Mô tả |
|-------|---------|-------|
| `parking/gate/gate01/in` | `{"card_id": "..."}` | Xe vào (quẹt RFID-A) |
| `parking/gate/gate01/out` | `{"card_id": "..."}` | Xe ra (quẹt RFID-B) |
| `parking/gate/gate01/heartbeat` | `{"mac": "...", "ip": "..."}` | Heartbeat |
| `parking/gate/gate01/status` | `{"online": true}` | Status |

### **ESP32 THANHTOAN ↔️ App**
| Direction | Topic | Payload | Mô tả |
|-----------|-------|---------|-------|
| ESP32 → App | `parking/payment/gate02/card_scanned` | `{"card_id": "..."}` | Quẹt thẻ tại terminal |
| **App → ESP32** | `parking/payment/gate02/vehicle_info` | `{"plate": "...", "fee": 3000, "paid": false}` | Gửi info xe về terminal |
| ESP32 → App | `parking/payment/gate02/payment_confirmed` | `{"card_id": "..."}` | Xác nhận đã thanh toán |
| ESP32 → App | `parking/payment/gate02/heartbeat` | `{"mac": "...", "ip": "..."}` | Heartbeat |

---

## 📋 Bảng tóm tắt Payment Terminal Logic

| Trường hợp | `paid` | `fee` | Terminal hiển thị | Revenue thay đổi | Ghi chú |
|------------|--------|-------|-------------------|------------------|---------|
| **Lần đầu quẹt** (chưa trả) | `false` | `3000` | ⚪ Xanh lá: "SỐ TIỀN: 3000 VND" + countdown | +3000 | Cho phép thanh toán |
| **Quẹt lại** (đã trả) | `true` | `0` | 🟠 Cam: "ĐÃ THANH TOÁN RỒI!" | Không đổi | Không cho thanh toán lại |
| **Xe ra** (đã trả tại terminal) | `true` | `0` | N/A (cổng ra) | Không đổi | Hiển thị "Đã thanh toán" |
| **Xe ra** (chưa trả) | `false` | `3000` | N/A (cổng ra) | +3000 | Thu phí tại cổng |

### **Logic ESP32 thanhtoan:**
```cpp
// Dòng 696-708 trong main.cpp
if (currentVehicle.isPaid && currentVehicle.fee == 0) {
    // ✅ CẢ HAI điều kiện phải đúng:
    // 1. paid = true
    // 2. fee = 0
    displayAlreadyPaid();  // Hiển thị màn hình CAM
} else {
    // Hiển thị màn hình thanh toán bình thường
    displayVehicleInfo();
}
```

---

## ⚠️ LƯU Ý QUAN TRỌNG

### **1. Payment Terminal KHÔNG PHẢI cổng ra**
- ❌ Terminal không xóa xe khỏi in_records
- ✅ Terminal chỉ đánh dấu `paid = true`
- ✅ Xe vẫn trong bãi, chờ quẹt RFID-B để ra

### **2. Revenue chỉ tính 1 lần**
- Nếu thanh toán tại terminal → Revenue tăng ngay
- Khi ra, check `paid = true` → Không tính lại revenue
- Nếu không thanh toán → Thu phí tại cổng ra

### **3. Slot count**
- Slot chỉ giải phóng khi quẹt RFID-B (cổng ra)
- Thanh toán tại terminal KHÔNG giảm slot

---

## 🎯 Quy trình đầy đủ

**Khách hàng GỬI XE:**
1. Quẹt thẻ tại RFID-A (vào) → Chụp ảnh → Lưu biển số
2. Xe đỗ trong bãi

**Khách hàng THANH TOÁN (Tùy chọn):**
3a. Quẹt thẻ tại Payment Terminal → Hiển thị thông tin → Auto thanh toán → Revenue tăng
3b. Hoặc không thanh toán, trả tiền khi ra

**Khách hàng LẤY XE:**
4. Quẹt thẻ tại RFID-B (ra) → Chụp ảnh
5. Nếu đã thanh toán → Cho ra, không thu phí
6. Nếu chưa thanh toán → Thu phí, tăng revenue

---

**✅ Hệ thống hoàn chỉnh!**

_Generated: 2025-10-18_
