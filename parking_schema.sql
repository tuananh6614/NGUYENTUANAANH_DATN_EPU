-- ================================================
-- PARKING SYSTEM DATABASE SCHEMA (SIMPLIFIED VERSION)
-- Version: 1.0.2
-- Date: 2025-10-22
-- Description: Chỉ tạo database và các bảng chính, không có dữ liệu mẫu, views, procedures hoặc triggers.
-- ================================================

-- Tạo database nếu chưa tồn tại
CREATE DATABASE IF NOT EXISTS parking_system
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- Sử dụng database parking_system
USE parking_system;

-- ================================================
-- TABLE: vehicles_in
-- Mô tả: Lưu thông tin các xe đang trong bãi
-- ================================================
DROP TABLE IF EXISTS vehicles_in;
CREATE TABLE vehicles_in (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate VARCHAR(20) COMMENT 'Biển số xe',
    rfid VARCHAR(20) UNIQUE NOT NULL COMMENT 'Mã thẻ RFID',
    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Thời gian vào',
    entry_image TEXT COMMENT 'Đường dẫn ảnh lúc vào',
    gate_id VARCHAR(20) COMMENT 'ID cổng vào',
    INDEX idx_rfid (rfid),
    INDEX idx_plate (plate),
    INDEX idx_entry_time (entry_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Xe đang trong bãi';

-- ================================================
-- TABLE: vehicles_out
-- Mô tả: Lịch sử các xe đã ra khỏi bãi
-- ================================================
DROP TABLE IF EXISTS vehicles_out;
CREATE TABLE vehicles_out (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate VARCHAR(20) COMMENT 'Biển số xe',
    rfid VARCHAR(20) COMMENT 'Mã thẻ RFID',
    entry_time DATETIME COMMENT 'Thời gian vào',
    exit_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Thời gian ra',
    entry_image TEXT COMMENT 'Đường dẫn ảnh lúc vào',
    exit_image TEXT COMMENT 'Đường dẫn ảnh lúc ra',
    gate_id VARCHAR(20) COMMENT 'ID cổng ra',
    duration_minutes INT COMMENT 'Thời gian gửi (phút)',
    fee DECIMAL(10,2) COMMENT 'Phí gửi xe (VNĐ)',
    INDEX idx_rfid (rfid),
    INDEX idx_exit_time (exit_time),
    INDEX idx_plate (plate),
    INDEX idx_date (DATE(exit_time))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Lịch sử xe ra';

-- ================================================
-- TABLE: rfid_cards
-- Mô tả: Quản lý thông tin thẻ RFID
-- ================================================
DROP TABLE IF EXISTS rfid_cards;
CREATE TABLE rfid_cards (
    rfid VARCHAR(20) PRIMARY KEY COMMENT 'Mã thẻ RFID',
    plate VARCHAR(20) COMMENT 'Biển số xe đăng ký',
    card_type ENUM('monthly', 'prepaid', 'visitor') DEFAULT 'visitor' COMMENT 'Loại thẻ',
    balance DECIMAL(10,2) DEFAULT 0 COMMENT 'Số dư (VNĐ)',
    status ENUM('active', 'blocked', 'expired') DEFAULT 'active' COMMENT 'Trạng thái',
    registered_date DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Ngày đăng ký',
    expiry_date DATETIME COMMENT 'Ngày hết hạn (cho thẻ tháng)',
    owner_name VARCHAR(100) COMMENT 'Tên chủ thẻ',
    phone VARCHAR(20) COMMENT 'Số điện thoại',
    notes TEXT COMMENT 'Ghi chú',
    INDEX idx_plate (plate),
    INDEX idx_status (status),
    INDEX idx_card_type (card_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Quản lý thẻ RFID';

-- ================================================
-- TABLE: transactions
-- Mô tả: Lịch sử giao dịch nạp tiền/trừ tiền
-- ================================================
DROP TABLE IF EXISTS transactions;
CREATE TABLE transactions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rfid VARCHAR(20) NOT NULL COMMENT 'Mã thẻ RFID',
    amount DECIMAL(10,2) NOT NULL COMMENT 'Số tiền (+/-)',
    type ENUM('charge', 'deduct', 'refund') DEFAULT 'deduct' COMMENT 'Loại giao dịch',
    balance_after DECIMAL(10,2) COMMENT 'Số dư sau giao dịch',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Thời gian',
    gate_id VARCHAR(20) COMMENT 'ID cổng (nếu liên quan)',
    description TEXT COMMENT 'Mô tả',
    operator VARCHAR(50) COMMENT 'Người thực hiện',
    INDEX idx_rfid (rfid),
    INDEX idx_timestamp (timestamp),
    INDEX idx_type (type),
    INDEX idx_date (DATE(timestamp)),
    FOREIGN KEY (rfid) REFERENCES rfid_cards(rfid) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Lịch sử giao dịch';

-- ================================================
-- TABLE: gates
-- Mô tả: Quản lý thông tin các cổng/thiết bị
-- ================================================
DROP TABLE IF EXISTS gates;
CREATE TABLE gates (
    gate_id VARCHAR(20) PRIMARY KEY COMMENT 'ID cổng',
    name VARCHAR(50) COMMENT 'Tên cổng',
    ip VARCHAR(15) COMMENT 'Địa chỉ IP',
    firmware_version VARCHAR(20) COMMENT 'Phiên bản firmware',
    last_seen DATETIME COMMENT 'Lần online cuối',
    status ENUM('online', 'offline', 'error') DEFAULT 'offline' COMMENT 'Trạng thái',
    location VARCHAR(100) COMMENT 'Vị trí',
    device_type ENUM('thanhtoan', 'baidoxe', 'camera') COMMENT 'Loại thiết bị',
    notes TEXT COMMENT 'Ghi chú',
    INDEX idx_status (status),
    INDEX idx_last_seen (last_seen)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Quản lý thiết bị gates';

-- ================================================
-- TABLE: firmware_versions
-- Mô tả: Quản lý các phiên bản firmware
-- ================================================
DROP TABLE IF EXISTS firmware_versions;
CREATE TABLE firmware_versions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    version VARCHAR(20) UNIQUE NOT NULL COMMENT 'Phiên bản',
    file_name VARCHAR(100) COMMENT 'Tên file .bin',
    file_path TEXT COMMENT 'Đường dẫn file',
    file_size INT COMMENT 'Kích thước file (bytes)',
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Ngày upload',
    changelog TEXT COMMENT 'Thay đổi',
    compatible_devices TEXT COMMENT 'Thiết bị tương thích (thanhtoan, baidoxe)',
    md5_checksum VARCHAR(32) COMMENT 'MD5 checksum để verify',
    is_stable BOOLEAN DEFAULT FALSE COMMENT 'Phiên bản ổn định?',
    download_count INT DEFAULT 0 COMMENT 'Số lần download',
    INDEX idx_version (version),
    INDEX idx_upload_date (upload_date),
    INDEX idx_is_stable (is_stable)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Quản lý firmware';

-- ================================================
-- TABLE: system_logs
-- Mô tả: Log hệ thống
-- ================================================
DROP TABLE IF EXISTS system_logs;
CREATE TABLE system_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Thời gian',
    level ENUM('INFO', 'WARNING', 'ERROR', 'CRITICAL') DEFAULT 'INFO' COMMENT 'Mức độ',
    component VARCHAR(50) COMMENT 'Thành phần (ALPR, MQTT, DB, OTA)',
    message TEXT COMMENT 'Nội dung log',
    gate_id VARCHAR(20) COMMENT 'ID cổng liên quan',
    details TEXT COMMENT 'Chi tiết bổ sung (JSON format as TEXT)',
    INDEX idx_timestamp (timestamp),
    INDEX idx_level (level),
    INDEX idx_gate (gate_id),
    INDEX idx_component (component),
    INDEX idx_date (DATE(timestamp))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='System logs';

-- ================================================
-- TABLE: parking_config
-- Mô tả: Cấu hình hệ thống (phí, thời gian, etc)
-- ================================================
DROP TABLE IF EXISTS parking_config;
CREATE TABLE parking_config (
    config_key VARCHAR(50) PRIMARY KEY COMMENT 'Tên config',
    config_value TEXT COMMENT 'Giá trị',
    data_type ENUM('string', 'number', 'boolean', 'json') DEFAULT 'string' COMMENT 'Kiểu dữ liệu',
    description TEXT COMMENT 'Mô tả',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Cập nhật lần cuối',
    updated_by VARCHAR(50) COMMENT 'Người cập nhật'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Cấu hình hệ thống';

-- ================================================
-- VERIFICATION QUERIES
-- ================================================

-- Check tables
SELECT 'Tables created:' as Status;
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'parking_system' 
AND table_type = 'BASE TABLE';

-- Final status
SELECT 'Database tables setup completed successfully!' as final_status;