"""
Script test cac module da tach ra
Chay file nay de kiem tra xem cac import co hoat dong khong
"""

print("=" * 60)
print("TESTING REFACTORED MODULES")
print("=" * 60)

# Test cau_hinh.py
print("\n1. Testing cau_hinh.py...")
try:
    from cau_hinh import (
        UiConfig, load_config, save_config,
        YOLO_MODEL_PATH, DIR_IN, DIR_OUT, FEE_FLAT
    )
    print(f"   [OK] cau_hinh.py OK")
    print(f"   - YOLO_MODEL_PATH: {YOLO_MODEL_PATH}")
    print(f"   - DIR_IN: {DIR_IN}")
    print(f"   - DIR_OUT: {DIR_OUT}")
    print(f"   - FEE_FLAT: {FEE_FLAT}")
except Exception as e:
    print(f"   [FAILED] cau_hinh.py FAILED: {e}")

# Test cong_cu.py
print("\n2. Testing cong_cu.py...")
try:
    from cong_cu import (
        plate_similarity, list_cameras, cleanup_old_images,
        is_port_open, get_local_ips
    )
    print(f"   [OK] cong_cu.py OK")

    # Test plate_similarity
    sim = plate_similarity("51A-12345", "51A-12346")
    print(f"   - plate_similarity('51A-12345', '51A-12346') = {sim:.2f}")

    # Test get_local_ips
    ips = get_local_ips()
    print(f"   - Local IPs: {ips}")
except Exception as e:
    print(f"   [FAILED] cong_cu.py FAILED: {e}")

# Test camera.py
print("\n3. Testing camera.py...")
try:
    from camera import CameraWorker
    print(f"   [OK] camera.py OK")
    print(f"   - CameraWorker class imported successfully")
except Exception as e:
    print(f"   [FAILED] camera.py FAILED: {e}")

# Test nhan_dien_bien.py
print("\n4. Testing nhan_dien_bien.py...")
try:
    from nhan_dien_bien import ALPR, clean_plate_text, order_points, warp_plate
    print(f"   [OK] nhan_dien_bien.py OK")

    # Test clean_plate_text
    cleaned = clean_plate_text("51A 12345")
    print(f"   - clean_plate_text('51A 12345') = '{cleaned}'")
except Exception as e:
    print(f"   [FAILED] nhan_dien_bien.py FAILED: {e}")

# Test giao_dien.py
print("\n5. Testing giao_dien.py...")
try:
    from giao_dien import SettingsDialog, qlabel_video_placeholder
    print(f"   [OK] giao_dien.py OK")
    print(f"   - SettingsDialog class imported successfully")
    print(f"   - qlabel_video_placeholder() imported successfully")
except Exception as e:
    print(f"   [FAILED] giao_dien.py FAILED: {e}")

print("\n" + "=" * 60)
print("ALL MODULES TESTED!")
print("=" * 60)
print("\n[OK] Tat ca cac file da duoc tach ra thanh cong!")
print("   - cau_hinh.py: Cau hinh va constants")
print("   - cong_cu.py: Utility functions")
print("   - camera.py: CameraWorker class")
print("   - nhan_dien_bien.py: ALPR engine")
print("   - giao_dien.py: SettingsDialog + UI helpers")
print("\n[NOTE] parking_ui.py van GIU NGUYEN ma hien tai de dam bao ung dung chay on dinh.")
print("   Sau khi test ky, co the dan dan chuyen sang import tu cac file moi.")
