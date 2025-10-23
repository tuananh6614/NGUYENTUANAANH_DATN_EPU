-- ================================================
-- LỆNH KIỂM TRA DATABASE PARKING_SYSTEM
-- Server: MariaDB 10.4.32 tại 192.168.1.165
-- User: root (no password)
-- ================================================

-- 1. XEM TẤT CẢ CÁC DATABASE
SHOW DATABASES;

-- 2. CHỌN DATABASE PARKING_SYSTEM
USE parking_system;

-- 3. XEM TẤT CẢ CÁC BẢNG
SHOW TABLES;

-- 4. XEM CẤU TRÚC CỦA TỪNG BẢNG
DESCRIBE vehicles_in;
DESCRIBE vehicles_out;
DESCRIBE rfid_cards;
DESCRIBE transactions;
DESCRIBE gates;
DESCRIBE firmware_versions;
DESCRIBE system_logs;
DESCRIBE parking_config;

-- 5. XEM DỮ LIỆU TRONG CÁC BẢNG
SELECT * FROM vehicles_in;
SELECT * FROM vehicles_out;
SELECT * FROM rfid_cards;
SELECT * FROM transactions;
SELECT * FROM gates;
SELECT * FROM firmware_versions;
SELECT * FROM system_logs;
SELECT * FROM parking_config;

-- 6. ĐẾM SỐ LƯỢNG RECORDS
SELECT 'vehicles_in' as Table_Name, COUNT(*) as Record_Count FROM vehicles_in
UNION ALL
SELECT 'vehicles_out', COUNT(*) FROM vehicles_out
UNION ALL
SELECT 'rfid_cards', COUNT(*) FROM rfid_cards
UNION ALL
SELECT 'transactions', COUNT(*) FROM transactions
UNION ALL
SELECT 'gates', COUNT(*) FROM gates
UNION ALL
SELECT 'firmware_versions', COUNT(*) FROM firmware_versions
UNION ALL
SELECT 'system_logs', COUNT(*) FROM system_logs
UNION ALL
SELECT 'parking_config', COUNT(*) FROM parking_config;

-- 7. XEM THÔNG TIN CHI TIẾT VỀ TỪNG BẢNG
SELECT
    TABLE_NAME,
    TABLE_ROWS,
    ROUND(((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024), 2) AS Size_MB,
    ENGINE,
    TABLE_COLLATION
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'parking_system'
AND TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_NAME;
