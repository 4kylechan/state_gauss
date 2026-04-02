from ultralytics import YOLO
import cv2
from collections import defaultdict
import pygame
import time
import threading

# YOLO modelini yükle
model = YOLO("yolov8n.pt")

# Kamera aç
cap = cv2.VideoCapture(0)

# pygame ses sistemi başlat
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.set_num_channels(8)
print("pygame mixer hazir")

# State tutma
zone_counts = defaultdict(int)
last_zone = None

# Ses cooldown
last_sound_time = 0
COOLDOWN = 0.5

# COCO class id: cell phone = 67
PHONE_CLASS_ID = 67

# Zone bazli temel sesler
base_sound_map = {
    "sol_ust": pygame.mixer.Sound("sounds/sol_ust.wav"),
    "sag_ust": pygame.mixer.Sound("sounds/sag_ust.wav"),
    "sol_alt": pygame.mixer.Sound("sounds/sol_alt.wav"),
    "sag_alt": pygame.mixer.Sound("sounds/sag_alt.wav")
}

def play_sound(zone, count):
    global last_sound_time
    now = time.time()
    if now - last_sound_time < COOLDOWN:
        return

    def _play():
        try:
            channel = pygame.mixer.find_channel(True)
            if channel:
                channel.play(base_sound_map[zone])
                print(f"SES: {zone} ({count})")
            else:
                print("Uyarı: boş ses kanalı bulunamadı")
        except Exception as e:
            print("Ses hatasi:", e)

    threading.Thread(target=_play, daemon=True).start()
    last_sound_time = now

def get_zone(cx, cy, w, h):
    mid_x = w // 2
    mid_y = h // 2
    if cx < mid_x and cy < mid_y:
        return "sol_ust"
    elif cx >= mid_x and cy < mid_y:
        return "sag_ust"
    elif cx < mid_x and cy >= mid_y:
        return "sol_alt"
    else:
        return "sag_alt"

frame_count = 0
event_text = ""   # yorum metni global olarak tutulacak

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera okunamadi.")
        break

    h, w, _ = frame.shape
    mid_x = w // 2
    mid_y = h // 2

    # Zone çizgileri
    cv2.line(frame, (mid_x, 0), (mid_x, h), (0, 255, 0), 2)
    cv2.line(frame, (0, mid_y), (w, mid_y), (0, 255, 0), 2)

    # Zone isimleri
    cv2.putText(frame, "SOL UST", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SAG UST", (mid_x + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SOL ALT", (20, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SAG ALT", (mid_x + 20, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # YOLO inference'i her 2 karede bir çalıştır
    if frame_count % 2 == 0:
        results = model(frame)
    frame_count += 1

    detected_phone = False
    current_zone = None

    if frame_count % 2 == 0:
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls == PHONE_CLASS_ID and conf > 0.40:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                current_zone = get_zone(cx, cy, w, h)
                detected_phone = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                cv2.putText(frame,
                            f"Telefon | {current_zone} | conf:{conf:.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2)
                break

    if detected_phone:
        if current_zone != last_zone:
            zone_counts[current_zone] += 1
            count = zone_counts[current_zone]

            if last_zone is None:
                event_text = f"Telefon ilk kez {current_zone.upper()} bolgesinde goruldu"
                print(f"EVENT: {event_text}")
            else:
                event_text = f"Telefon {last_zone.upper()} -> {current_zone.upper()} bolgesine gecti"
                print(f"EVENT: {event_text}")

            print(f"STATE: telefon {count} defa {current_zone} bolgede goruldu")
            play_sound(current_zone, count)
            last_zone = current_zone

        y_offset = 60
        for zone_name in ["sol_ust", "sag_ust", "sol_alt", "sag_alt"]:
            text = f"{zone_name}: {zone_counts[zone_name]}"
            cv2.putText(frame, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            y_offset += 30

        cv2.putText(frame, f"Anlik Zone: {last_zone}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Telefon tespit edilmedi",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- ORTADA yorum metni HER ZAMAN çiz ---
    if event_text:
        cv2.putText(frame, event_text,
                    (w // 2 - 250, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),   # siyah renk
                    2,
                    cv2.LINE_AA)

    cv2.imshow("PlayX Zone + Event + State + Audio", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
