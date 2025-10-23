#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script để xem dữ liệu trong database parking_system
Server: MariaDB 10.4.32 tại 192.168.1.165
User: root (no password)
"""

import mysql.connector
import sys
import json
from datetime import datetime

# Set UTF-8 encoding for console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def connect_db():
    """Kết nối tới database"""
    return mysql.connector.connect(
        host='192.168.1.165',
        port=3306,
        user='root',
        password='',
        database='parking_system',
        charset='utf8mb4'
    )

def show_tables():
    """Hiển thị danh sách các bảng"""
    conn = connect_db()
    cursor = conn.cursor()

    print("=" * 60)
    print("DANH SÁCH CÁC BẢNG TRONG DATABASE 'parking_system'")
    print("=" * 60)

    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    for i, table in enumerate(tables, 1):
        print(f"{i}. {table[0]}")

    cursor.close()
    conn.close()
    print()

def show_table_info():
    """Hiển thị thông tin chi tiết về các bảng"""
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    print("=" * 80)
    print("THÔNG TIN CHI TIẾT CÁC BẢNG")
    print("=" * 80)

    query = """
    SELECT
        TABLE_NAME,
        TABLE_ROWS,
        ROUND(((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024), 2) AS Size_MB,
        ENGINE,
        TABLE_COLLATION
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = 'parking_system'
    AND TABLE_TYPE = 'BASE TABLE'
    ORDER BY TABLE_NAME
    """

    cursor.execute(query)
    tables = cursor.fetchall()

    print(f"{'Bảng':<25} {'Số dòng':>10} {'Kích thước':>12} {'Engine':>10}")
    print("-" * 80)

    for table in tables:
        print(f"{table['TABLE_NAME']:<25} {table['TABLE_ROWS']:>10} {table['Size_MB']:>10} MB {table['ENGINE']:>10}")

    cursor.close()
    conn.close()
    print()

def show_data_counts():
    """Đếm số lượng records trong mỗi bảng"""
    conn = connect_db()
    cursor = conn.cursor()

    print("=" * 60)
    print("SỐ LƯỢNG DỮ LIỆU TRONG TỪNG BẢNG")
    print("=" * 60)

    tables = [
        'vehicles_in',
        'vehicles_out',
        'rfid_cards',
        'transactions',
        'gates',
        'firmware_versions',
        'system_logs',
        'parking_config'
    ]

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table:<25} {count:>5} records")

    cursor.close()
    conn.close()
    print()

def show_rfid_cards():
    """Hiển thị danh sách thẻ RFID"""
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    print("=" * 100)
    print("DANH SÁCH THẺ RFID")
    print("=" * 100)

    cursor.execute("SELECT * FROM rfid_cards")
    cards = cursor.fetchall()

    if not cards:
        print("Chưa có thẻ RFID nào")
    else:
        print(f"{'RFID':<12} {'Biển số':<12} {'Loại thẻ':<10} {'Số dư':>15} {'Trạng thái':<10} {'Chủ thẻ':<20}")
        print("-" * 100)
        for card in cards:
            balance = f"{float(card['balance']):,.0f} đ"
            print(f"{card['rfid']:<12} {card['plate'] or 'N/A':<12} {card['card_type']:<10} {balance:>15} {card['status']:<10} {card['owner_name'] or 'N/A':<20}")

    cursor.close()
    conn.close()
    print()

def show_gates():
    """Hiển thị danh sách thiết bị/cổng"""
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    print("=" * 100)
    print("DANH SÁCH THIẾT BỊ/CỔNG")
    print("=" * 100)

    cursor.execute("SELECT * FROM gates")
    gates = cursor.fetchall()

    if not gates:
        print("Chưa có thiết bị nào")
    else:
        print(f"{'Gate ID':<12} {'Tên':<20} {'IP':<15} {'Loại':<12} {'Trạng thái':<10} {'Last Seen':<20}")
        print("-" * 100)
        for gate in gates:
            last_seen = gate['last_seen'].strftime('%Y-%m-%d %H:%M:%S') if gate['last_seen'] else 'Never'
            print(f"{gate['gate_id']:<12} {gate['name'] or 'N/A':<20} {gate['ip'] or 'N/A':<15} {gate['device_type'] or 'N/A':<12} {gate['status']:<10} {last_seen:<20}")

    cursor.close()
    conn.close()
    print()

def show_vehicles_in():
    """Hiển thị xe đang trong bãi"""
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    print("=" * 100)
    print("XE ĐANG TRONG BÃI")
    print("=" * 100)

    cursor.execute("SELECT * FROM vehicles_in ORDER BY entry_time DESC")
    vehicles = cursor.fetchall()

    if not vehicles:
        print("Hiện không có xe trong bãi")
    else:
        print(f"{'ID':<5} {'Biển số':<12} {'RFID':<12} {'Thời gian vào':<20} {'Gate ID':<10}")
        print("-" * 100)
        for v in vehicles:
            entry_time = v['entry_time'].strftime('%Y-%m-%d %H:%M:%S') if v['entry_time'] else 'N/A'
            print(f"{v['id']:<5} {v['plate'] or 'N/A':<12} {v['rfid']:<12} {entry_time:<20} {v['gate_id'] or 'N/A':<10}")

    cursor.close()
    conn.close()
    print()

def show_recent_exits():
    """Hiển thị lịch sử xe ra gần đây"""
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    print("=" * 120)
    print("LỊCH SỬ XE RA (10 GẦN NHẤT)")
    print("=" * 120)

    cursor.execute("""
        SELECT * FROM vehicles_out
        ORDER BY exit_time DESC
        LIMIT 10
    """)
    vehicles = cursor.fetchall()

    if not vehicles:
        print("Chưa có lịch sử xe ra")
    else:
        print(f"{'Biển số':<12} {'RFID':<12} {'Vào':<20} {'Ra':<20} {'Phút':>6} {'Phí':>12}")
        print("-" * 120)
        for v in vehicles:
            entry = v['entry_time'].strftime('%Y-%m-%d %H:%M:%S') if v['entry_time'] else 'N/A'
            exit_t = v['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if v['exit_time'] else 'N/A'
            fee = f"{float(v['fee'] or 0):,.0f} đ"
            print(f"{v['plate'] or 'N/A':<12} {v['rfid'] or 'N/A':<12} {entry:<20} {exit_t:<20} {v['duration_minutes'] or 0:>6} {fee:>12}")

    cursor.close()
    conn.close()
    print()

def show_today_stats():
    """Thống kê hôm nay"""
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    print("=" * 60)
    print("THỐNG KÊ HÔM NAY")
    print("=" * 60)

    # Tổng xe vào hôm nay
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM vehicles_out
        WHERE DATE(exit_time) = CURDATE()
    """)
    today_count = cursor.fetchone()['count']

    # Tổng doanh thu hôm nay
    cursor.execute("""
        SELECT COALESCE(SUM(fee), 0) as revenue
        FROM vehicles_out
        WHERE DATE(exit_time) = CURDATE()
    """)
    today_revenue = cursor.fetchone()['revenue']

    # Xe đang trong bãi
    cursor.execute("SELECT COUNT(*) as count FROM vehicles_in")
    in_count = cursor.fetchone()['count']

    print(f"Xe đã ra hôm nay:        {today_count:>5} xe")
    print(f"Doanh thu hôm nay:       {float(today_revenue):>12,.0f} đ")
    print(f"Xe đang trong bãi:       {in_count:>5} xe")

    cursor.close()
    conn.close()
    print()

def main():
    """Main function"""
    try:
        print("\n")
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 10 + "DATABASE PARKING_SYSTEM VIEWER" + " " * 18 + "║")
        print("║" + " " * 10 + "Server: MariaDB 10.4.32" + " " * 25 + "║")
        print("║" + " " * 10 + "Host: 192.168.1.165" + " " * 29 + "║")
        print("╚" + "═" * 58 + "╝")
        print()

        # Hiển thị thông tin
        show_tables()
        show_table_info()
        show_data_counts()
        show_today_stats()
        show_rfid_cards()
        show_gates()
        show_vehicles_in()
        show_recent_exits()

        print("=" * 60)
        print("HOÀN TẤT!")
        print("=" * 60)

    except mysql.connector.Error as e:
        print(f"❌ Lỗi kết nối database: {e}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
