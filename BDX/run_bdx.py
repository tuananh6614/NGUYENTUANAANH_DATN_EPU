"""
Script tự động restart app khi có thay đổi file - Dành cho BDX
Sử dụng watchdog để theo dõi thay đổi file
"""
import sys
import subprocess
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        GREEN = YELLOW = CYAN = RED = ""
    class Style:
        RESET_ALL = ""

APP_FILE = "doan_baidoxe.py"
DEBOUNCE_TIME = 3.0  # Đợi 3 giây trước khi restart
CAMERA_RELEASE_TIME = 2.0  # Đợi camera release

class Handler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_modified = 0

    def on_modified(self, event):
        if event.src_path.endswith(APP_FILE) or event.src_path.endswith(".py"):
            now = time.time()
            if now - self.last_modified < DEBOUNCE_TIME:
                print(Fore.YELLOW + f"[Watcher] Bỏ qua thay đổi (debounce: {DEBOUNCE_TIME}s)")
                return
            self.last_modified = now
            print(Fore.YELLOW + f"[Watcher] File Python thay đổi, restart app...")
            restart_app()

p = None

def start_app():
    global p
    if p is not None:
        print(Fore.YELLOW + "[Runner] App đang chạy, bỏ qua start_app()")
        return
    print(Fore.GREEN + "[Runner] Bắt đầu chạy BDX app...")
    p = subprocess.Popen([sys.executable, APP_FILE])
    print(Fore.GREEN + f"[Runner] App started với PID: {p.pid}")

def restart_app():
    """Restart app với proper cleanup"""
    global p
    if p is not None:
        print(Fore.CYAN + "[Runner] Dừng app cũ...")
        p.terminate()

        try:
            p.wait(timeout=5)
            print(Fore.GREEN + "[Runner] App đã dừng sạch")
        except subprocess.TimeoutExpired:
            print(Fore.RED + "[Runner] App không dừng kịp, kill...")
            p.kill()
            p.wait()
            print(Fore.RED + "[Runner] App đã bị kill")

        # Đợi camera release hoàn toàn
        print(Fore.YELLOW + f"[Runner] Đợi {CAMERA_RELEASE_TIME}s để camera release...")
        time.sleep(CAMERA_RELEASE_TIME)

    print(Fore.GREEN + "[Runner] Chạy lại app...")
    p = subprocess.Popen([sys.executable, APP_FILE])
    print(Fore.GREEN + f"[Runner] App restarted với PID: {p.pid}")

def stop_app():
    """Clean shutdown"""
    global p
    if p is not None:
        print(Fore.CYAN + "[Runner] Tắt app...")
        p.terminate()
        try:
            p.wait(timeout=5)
            print(Fore.GREEN + "[Runner] App đã tắt")
        except subprocess.TimeoutExpired:
            print(Fore.RED + "[Runner] App không dừng, kill...")
            p.kill()
            p.wait()
            print(Fore.RED + "[Runner] App đã bị kill")

if __name__ == "__main__":
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + "  BDX APP AUTO-RELOADER")
    print(Fore.CYAN + f"  File: {APP_FILE}")
    print(Fore.CYAN + f"  Debounce: {DEBOUNCE_TIME}s | Camera wait: {CAMERA_RELEASE_TIME}s")
    print(Fore.CYAN + "=" * 60)

    if not os.path.exists(APP_FILE):
        print(Fore.RED + f"[Error] Không tìm thấy file {APP_FILE}")
        print(Fore.YELLOW + f"[Hint] Đảm bảo bạn đang ở trong folder BDX/")
        sys.exit(1)

    start_app()

    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    print(Fore.GREEN + "[Watcher] Đang theo dõi thay đổi file Python trong folder...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n[Watcher] Nhận Ctrl+C, đang dừng...")
        observer.stop()
        stop_app()
        print(Fore.GREEN + "[Watcher] Đã dừng sạch")

    observer.join()
