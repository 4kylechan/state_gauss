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

zone_counts = defaultdict(int)
last_zone = None
last_sound_time = 0
COOLDOWN = 0.5
PHONE_CLASS_ID = 67

# Ses dosyalarını bir kez yükleyelim (Sürekli diskten okumaması için)
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
    except Exception as e:
        print(f"Ses çalma hatası: {e}")
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

    # Zone İsimleri
    cv2.putText(frame, "SOL UST", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SAG UST", (mid_x + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SOL ALT", (20, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SAG ALT", (mid_x + 20, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- YOLO TESPİT ---
    # Her 2 karede bir çalıştır ama results'ı her zaman kontrol et
    if frame_count % 2 == 0:
        results = model.predict(frame, conf=0.40, classes=[PHONE_CLASS_ID], verbose=False)
    frame_count += 1

    detected_phone = False
    current_zone = None

    if results and len(results[0].boxes) > 0:
        # En yüksek skorlu kutuyu al
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0].item())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        current_zone = get_zone(cx, cy, w, h)
        detected_phone = True

        # Kutuyu ve merkezi çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Telefon {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # --- MANTIK VE OLAY YÖNETİMİ ---
        if current_zone != last_zone:
            zone_counts[current_zone] += 1
            if last_zone is None:
                event_text = f"Telefon ILK KEZ {current_zone.upper()} bolgesinde"
            else:
                event_text = f"GECIS: {last_zone.upper()} -> {current_zone.upper()}"
            
            # Sesi ayrı thread'de çal
            threading.Thread(target=play_sound, args=(current_zone,), daemon=True).start()
            last_zone = current_zone

    # --- EKRAN BİLGİ TABLOSU (Sol Üst Köşe) ---
    y_offset = 60
    for z_name in ["sol_ust", "sag_ust", "sol_alt", "sag_alt"]:
        color = (0, 255, 255) if z_name == current_zone else (255, 255, 0)
        cv2.putText(frame, f"{z_name}: {zone_counts[z_name]}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25

    # --- DURUM VE OLAY METİNLERİ ---
    if not detected_phone:
        cv2.putText(frame, "TELEFON TESPIT EDILMEDI", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Anlik Zone: {current_zone}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if event_text:
        # Metni merkeze siyah gölgeli yaz
        cv2.putText(frame, event_text, (w // 2 - 200, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3) # Gölge
        cv2.putText(frame, event_text, (w // 2 - 200, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) # Ana metin

    cv2.imshow("Gelismiş Telefon Takip Sistemi", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()