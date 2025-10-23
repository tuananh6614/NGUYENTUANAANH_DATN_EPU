import mysql.connector
from mysql.connector import Error
from datetime import datetime
import logging

class ParkingDB:
    """MySQL Database handler for Parking System"""

    def __init__(self, host='192.168.1.165', port=3306, user='root', password='', database='parking_system'):
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
            'autocommit': True,
            'charset': 'utf8mb4'
        }
        self.conn = None
        self.connect()

    def connect(self):
        """Connect to MySQL database"""
        try:
            self.conn = mysql.connector.connect(**self.config)
            if self.conn.is_connected():
                logging.info(f"[DB] Connected to MySQL at {self.config['host']}")
                return True
        except Error as e:
            logging.error(f"[DB] Connection failed: {e}")
            return False

    def reconnect(self):
        """Reconnect if connection is lost"""
        if not self.conn or not self.conn.is_connected():
            logging.warning("[DB] Connection lost, reconnecting...")
            self.connect()

    def execute_query(self, query, params=None):
        """Execute a query with auto-reconnect"""
        self.reconnect()
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            cursor.close()
            return True
        except Error as e:
            logging.error(f"[DB] Query failed: {e}")
            return False

    def fetch_one(self, query, params=None):
        """Fetch one result"""
        self.reconnect()
        try:
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            result = cursor.fetchone()
            cursor.close()
            return result
        except Error as e:
            logging.error(f"[DB] Fetch failed: {e}")
            return None

    def fetch_all(self, query, params=None):
        """Fetch all results"""
        self.reconnect()
        try:
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            logging.error(f"[DB] Fetch failed: {e}")
            return []

    # ==================== VEHICLE OPERATIONS ====================

    def vehicle_entry(self, rfid, plate, gate_id, image_path=None):
        """Record vehicle entry"""
        sql = """INSERT INTO vehicles_in (rfid, plate, gate_id, entry_image)
                 VALUES (%s, %s, %s, %s)
                 ON DUPLICATE KEY UPDATE plate=%s, entry_time=CURRENT_TIMESTAMP, entry_image=%s"""
        success = self.execute_query(sql, (rfid, plate, gate_id, image_path, plate, image_path))
        if success:
            logging.info(f"[DB] Vehicle IN: {plate} ({rfid}) at {gate_id}")
        return success

    def vehicle_exit(self, rfid, plate, gate_id, fee, image_path=None):
        """Record vehicle exit and move to history"""
        # Get entry info
        entry = self.fetch_one("SELECT entry_time, entry_image FROM vehicles_in WHERE rfid=%s", (rfid,))

        if not entry:
            logging.warning(f"[DB] No entry found for RFID: {rfid}")
            return False

        entry_time = entry['entry_time']
        entry_image = entry['entry_image']
        exit_time = datetime.now()
        duration = int((exit_time - entry_time).total_seconds() / 60)

        # Insert to history
        sql = """INSERT INTO vehicles_out
                 (rfid, plate, entry_time, exit_time, entry_image, exit_image, gate_id, duration_minutes, fee)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        success = self.execute_query(sql, (rfid, plate, entry_time, exit_time, entry_image, image_path, gate_id, duration, fee))

        if success:
            # Remove from vehicles_in
            self.execute_query("DELETE FROM vehicles_in WHERE rfid=%s", (rfid,))
            logging.info(f"[DB] Vehicle OUT: {plate} - Duration: {duration}min - Fee: {fee}")

        return success

    def get_vehicle_in_parking(self, rfid):
        """Get vehicle info if it's in parking"""
        return self.fetch_one("SELECT * FROM vehicles_in WHERE rfid=%s", (rfid,))

    def get_all_vehicles_in_parking(self):
        """Get all vehicles currently in parking"""
        return self.fetch_all("SELECT * FROM vehicles_in ORDER BY entry_time DESC")

    # ==================== RFID CARD OPERATIONS ====================

    def get_card_info(self, rfid):
        """Get RFID card information"""
        return self.fetch_one("SELECT * FROM rfid_cards WHERE rfid=%s", (rfid,))

    def register_card(self, rfid, plate, card_type='visitor', balance=0):
        """Register a new RFID card"""
        sql = """INSERT INTO rfid_cards (rfid, plate, card_type, balance, status)
                 VALUES (%s, %s, %s, %s, 'active')
                 ON DUPLICATE KEY UPDATE plate=%s, card_type=%s, balance=%s"""
        success = self.execute_query(sql, (rfid, plate, card_type, balance, plate, card_type, balance))
        if success:
            logging.info(f"[DB] Card registered: {rfid} - {plate} ({card_type})")
        return success

    def charge_card(self, rfid, amount, gate_id=None):
        """Charge money to card"""
        # Update balance
        self.execute_query("UPDATE rfid_cards SET balance = balance + %s WHERE rfid=%s", (amount, rfid))

        # Get new balance
        card = self.get_card_info(rfid)
        new_balance = card['balance'] if card else 0

        # Log transaction
        sql = """INSERT INTO transactions (rfid, amount, type, balance_after, gate_id, description)
                 VALUES (%s, %s, 'charge', %s, %s, 'Nap tien')"""
        self.execute_query(sql, (rfid, amount, new_balance, gate_id))

        logging.info(f"[DB] Card charged: {rfid} +{amount} -> {new_balance}")
        return new_balance

    def deduct_fee(self, rfid, fee, gate_id=None):
        """Deduct parking fee from card"""
        # Update balance
        self.execute_query("UPDATE rfid_cards SET balance = balance - %s WHERE rfid=%s", (fee, rfid))

        # Get new balance
        card = self.get_card_info(rfid)
        new_balance = card['balance'] if card else 0

        # Log transaction
        sql = """INSERT INTO transactions (rfid, amount, type, balance_after, gate_id, description)
                 VALUES (%s, %s, 'deduct', %s, %s, 'Phi gui xe')"""
        self.execute_query(sql, (rfid, fee, new_balance, gate_id))

        logging.info(f"[DB] Fee deducted: {rfid} -{fee} -> {new_balance}")
        return new_balance

    def get_card_transactions(self, rfid, limit=10):
        """Get transaction history for a card"""
        sql = "SELECT * FROM transactions WHERE rfid=%s ORDER BY timestamp DESC LIMIT %s"
        return self.fetch_all(sql, (rfid, limit))

    # ==================== GATE OPERATIONS ====================

    def update_gate_status(self, gate_id, firmware_version=None, ip=None, status='online', name=None):
        """Update gate device status"""
        sql = """INSERT INTO gates (gate_id, name, ip, firmware_version, last_seen, status)
                 VALUES (%s, %s, %s, %s, NOW(), %s)
                 ON DUPLICATE KEY UPDATE
                 name=COALESCE(%s, name),
                 ip=COALESCE(%s, ip),
                 firmware_version=COALESCE(%s, firmware_version),
                 last_seen=NOW(),
                 status=%s"""
        self.execute_query(sql, (gate_id, name, ip, firmware_version, status, name, ip, firmware_version, status))

    def get_gate_status(self, gate_id):
        """Get gate status"""
        return self.fetch_one("SELECT * FROM gates WHERE gate_id=%s", (gate_id,))

    def get_all_gates(self):
        """Get all gates"""
        return self.fetch_all("SELECT * FROM gates ORDER BY last_seen DESC")

    # ==================== FIRMWARE OPERATIONS ====================

    def add_firmware_version(self, version, file_name, file_path, file_size, changelog='', compatible_devices=''):
        """Add new firmware version"""
        sql = """INSERT INTO firmware_versions (version, file_name, file_path, file_size, changelog, compatible_devices)
                 VALUES (%s, %s, %s, %s, %s, %s)"""
        success = self.execute_query(sql, (version, file_name, file_path, file_size, changelog, compatible_devices))
        if success:
            logging.info(f"[DB] Firmware added: {version} ({file_size} bytes)")
        return success

    def get_latest_firmware(self):
        """Get latest firmware version"""
        return self.fetch_one("SELECT * FROM firmware_versions ORDER BY upload_date DESC LIMIT 1")

    def get_all_firmware_versions(self):
        """Get all firmware versions"""
        return self.fetch_all("SELECT * FROM firmware_versions ORDER BY upload_date DESC")

    # ==================== LOGGING ====================

    def log_system(self, level, component, message, gate_id=None):
        """Add system log"""
        sql = """INSERT INTO system_logs (level, component, message, gate_id)
                 VALUES (%s, %s, %s, %s)"""
        self.execute_query(sql, (level, component, message, gate_id))

    def get_recent_logs(self, limit=100, level=None):
        """Get recent logs"""
        if level:
            sql = "SELECT * FROM system_logs WHERE level=%s ORDER BY timestamp DESC LIMIT %s"
            return self.fetch_all(sql, (level, limit))
        else:
            sql = "SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT %s"
            return self.fetch_all(sql, (limit,))

    # ==================== STATISTICS ====================

    def get_today_revenue(self):
        """Get today's revenue"""
        result = self.fetch_one("SELECT SUM(fee) as total FROM vehicles_out WHERE DATE(exit_time) = CURDATE()")
        return result['total'] if result and result['total'] else 0

    def get_today_vehicles_count(self):
        """Get today's vehicle count"""
        result = self.fetch_one("SELECT COUNT(*) as cnt FROM vehicles_out WHERE DATE(exit_time) = CURDATE()")
        return result['cnt'] if result else 0

    def get_vehicles_in_count(self):
        """Get current vehicles in parking"""
        result = self.fetch_one("SELECT COUNT(*) as cnt FROM vehicles_in")
        return result['cnt'] if result else 0

    def close(self):
        """Close database connection"""
        if self.conn and self.conn.is_connected():
            self.conn.close()
            logging.info("[DB] Connection closed")


# Test connection
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = ParkingDB()

    # Test queries
    print("Vehicles in parking:", db.get_vehicles_in_count())
    print("Today revenue:", db.get_today_revenue())
    print("All gates:", db.get_all_gates())

    db.close()
