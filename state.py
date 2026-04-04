from ultralytics import YOLO
import cv2
from collections import defaultdict
import pygame
import time
import threading

# 1. Hazırlık ve Ayarlar
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

pygame.mixer.init()
pygame.mixer.set_num_channels(8)

# State ve Sayaç Tutma
zone_counts = defaultdict(int)
zone_start_times = {} # Her bölge için kronometre
last_zone = None
last_incremented_zone = None # Aynı bölgede tekrar tekrar artışı önlemek için
last_sound_time = 0
COOLDOWN = 0.5
PHONE_CLASS_ID = 67
REQUIRED_TIME = 3.0 # Onay için gereken süre (saniye)

base_sound_map = {
    "sol_ust": "sounds/sol_ust.wav",
    "sag_ust": "sounds/sag_ust.wav",
    "sol_alt": "sounds/sol_alt.wav",
    "sag_alt": "sounds/sag_alt.wav"
}

def play_sound(zone):
    global last_sound_time
    now = time.time()
    if now - last_sound_time < COOLDOWN:
        return
    try:
        sound = pygame.mixer.Sound(base_sound_map[zone])
        sound.play()
    except:
        pass
    last_sound_time = now

def get_zone(cx, cy, w, h):
    mid_x, mid_y = w // 2, h // 2
    if cy < mid_y:
        return "sol_ust" if cx < mid_x else "sag_ust"
    else:
        return "sol_alt" if cx < mid_x else "sag_alt"

frame_count = 0
event_text = ""
results = None

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    mid_x, mid_y = w // 2, h // 2

    # --- GÖRSEL KATMANLAR (Zonelar ve Çizgiler) ---
    cv2.line(frame, (mid_x, 0), (mid_x, h), (0, 255, 0), 2)
    cv2.line(frame, (0, mid_y), (w, mid_y), (0, 255, 0), 2)

    cv2.putText(frame, "SOL UST", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SAG UST", (mid_x + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SOL ALT", (20, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SAG ALT", (mid_x + 20, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- YOLO TESPİT (2 karede bir) ---
    if frame_count % 2 == 0:
        results = model.predict(frame, conf=0.40, classes=[PHONE_CLASS_ID], verbose=False)
    frame_count += 1

    detected_phone = False
    current_zone = None

    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0].item())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        current_zone = get_zone(cx, cy, w, h)
        detected_phone = True

        # Kutuyu çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # --- 3 SANİYE ONAY MANTIĞI ---
        if current_zone != last_zone:
            # Bölgeye yeni giriş yapıldı, kronometreyi sıfırla
            zone_start_times[current_zone] = time.time()
            last_zone = current_zone
            last_incremented_zone = None 
        else:
            # Aynı bölgede kalmaya devam ediyor
            elapsed = time.time() - zone_start_times.get(current_zone, time.time())
            
            if elapsed >= REQUIRED_TIME:
                if last_incremented_zone != current_zone:
                    zone_counts[current_zone] += 1
                    event_text = f"ONAYLANDI: {current_zone.upper()} (+1)"
                    threading.Thread(target=play_sound, args=(current_zone,), daemon=True).start()
                    last_incremented_zone = current_zone
            else:
                # 3 saniye dolana kadar geri sayımı göster
                rem = REQUIRED_TIME - elapsed
                cv2.putText(frame, f"Onaylanıyor: {rem:.1f}s", (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    else:
        # Telefon kameradan çıkarsa takibi sıfırla
        last_zone = None
        last_incremented_zone = None

    # --- BİLGİ TABLOSU VE DURUM ---
    y_offset = 60
    for z_name in ["sol_ust", "sag_ust", "sol_alt", "sag_alt"]:
        color = (0, 255, 255) if z_name == current_zone else (255, 255, 0)
        cv2.putText(frame, f"{z_name}: {zone_counts[z_name]}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25

    if not detected_phone:
        cv2.putText(frame, "TELEFON TESPIT EDILMEDI", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Anlik: {current_zone}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if event_text:
        cv2.putText(frame, event_text, (w // 2 - 200, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(frame, event_text, (w // 2 - 200, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("PlayX Zone Tracker (3s Delay Mode)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()