from ultralytics import YOLO
import cv2
from collections import defaultdict, deque
import pygame
import time
import threading
from datetime import datetime

# --- SAHNEYİ HAZIRLAYALIM ---
model = YOLO("yolov8n.pt") # Bizim keskin gözlü elemanı çağıralım
cap = cv2.VideoCapture(0)  # Kamerayı dürtelim

# Ses sistemi (Dj Pygame iş başında)
pygame.mixer.init()
pygame.mixer.set_num_channels(8)

# Hafıza Defterimiz
zone_counts = defaultdict(int)    # Hangi bölge kaç gol yedi?
zone_start_times = {}             # Telefon bölgeye ne zaman "merhaba" dedi?
event_logs = deque(maxlen=8)      # Olayların dedikodu listesi (Son 8 kayıt)

# Takip Elemanları
last_zone = None                  # Az önce neredeydi bu?
last_incremented_zone = None      # "Zaten saydım abi" kontrolü
PHONE_CLASS_ID = 67               # Telefonun gizli kimlik numarası
REQUIRED_TIME = 3.0               # Sabır testi: 3 saniye durması lazım

# Ses Arşivi
base_sound_map = {
    "sol_ust": "sounds/sol_ust.wav",
    "sag_ust": "sounds/sag_ust.wav",
    "sol_alt": "sounds/sol_alt.wav",
    "sag_alt": "sounds/sag_alt.wav"
}

# --- YARDIMCI ARKADAŞLAR ---

def play_sound(zone):
    """Bölgeye özel şarkımızı çalalım."""
    try:
        sound = pygame.mixer.Sound(base_sound_map[zone])
        sound.play()
    except:
        pass # Ses çıkmazsa can sağlığı...

def get_zone(cx, cy, w, h):
    """Telefon ekranın hangi mahallesinde? Onu bulur."""
    mid_x, mid_y = w // 2, h // 2
    if cy < mid_y:
        return "sol_ust" if cx < mid_x else "sag_ust"
    else:
        return "sol_alt" if cx < mid_x else "sag_alt"

# --- ASIL EĞLENCE BURADA BAŞLIYOR (ANA DÖNGÜ) ---

frame_count = 0
results = None

while True:
    ret, frame = cap.read()
    if not ret: break # Kamera küstüyse çıkalım

    h, w, _ = frame.shape
    mid_x, mid_y = w // 2, h // 2

    # Ekrana bölge sınırlarını çizelim (Sınır namustur!)
    cv2.line(frame, (mid_x, 0), (mid_x, h), (0, 255, 0), 2)
    cv2.line(frame, (0, mid_y), (w, mid_y), (0, 255, 0), 2)

    # Mahalleye isim verelim
    cv2.putText(frame, "SOL UST", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "SAG UST", (mid_x + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # YOLO'ya "Bir baksana ne var orada?" diyelim (Her 2 karede bir yorulmasın diye)
    if frame_count % 2 == 0:
        results = model.predict(frame, conf=0.40, classes=[PHONE_CLASS_ID], verbose=False)
    frame_count += 1

    detected_phone = False
    current_zone = None

    # Eğer eleman bir telefon gördüyse...
    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        current_zone = get_zone(cx, cy, w, h)
        detected_phone = True

        # Telefonun etrafına sarı bir kutu konduralım
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # --- 3 SANİYE SABIR TESTİ MANTIĞI ---
        if current_zone != last_zone:
            # Yeni bölgeye girdi! Kronometreyi başlat.
            zone_start_times[current_zone] = time.time()
            last_zone = current_zone
            last_incremented_zone = None # Yeni yer, yeni şans
        else:
            # Hala aynı yerde mi? Ne kadar süredir bekliyor?
            gecen = time.time() - zone_start_times.get(current_zone, time.time())
            
            if gecen >= REQUIRED_TIME:
                if last_incremented_zone != current_zone:
                    # Tamamdır, 3 saniye doldu! Sayacı artır.
                    zone_counts[current_zone] += 1
                    
                    # Dedikodu listesine (log) ekle
                    an = datetime.now().strftime("%H:%M:%S")
                    event_logs.appendleft(f"[{an}] {current_zone.upper()} Tiklandi!")
                    
                    # Şenlik başlasın, müziği ver!
                    threading.Thread(target=play_sound, args=(current_zone,), daemon=True).start()
                    last_incremented_zone = current_zone
            else:
                # Daha dolmadı, ekranda geri sayım yapalım (Heyecan olsun)
                kalan = REQUIRED_TIME - gecen
                cv2.putText(frame, f"Sabret: {kalan:.1f}s", (x1, y2 + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Telefon kaybolursa her şeyi sıfırla, sil baştan
        last_zone = None
        last_incremented_zone = None

    # --- TABLO VE LOG PANELİ (GÖRSEL ŞOV) ---

    # Sol tarafa skor tablosu
    y_skor = 60
    for bolge in ["sol_ust", "sag_ust", "sol_alt", "sag_alt"]:
        renk = (0, 255, 255) if bolge == current_zone else (255, 255, 0)
        cv2.putText(frame, f"{bolge}: {zone_counts[bolge]}", (20, y_skor), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)
        y_skor += 30

    # Sağ tarafa şeffaf "Neler Oldu?" paneli
    cv2.rectangle(frame, (w - 260, 0), (w, 280), (0, 0, 0), -1) # Arka plan
    cv2.putText(frame, " OLAYLAR", (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    y_log = 65
    for log in event_logs:
        cv2.putText(frame, log, (w - 250, y_log), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_log += 25

    # Alt kısma ufak bir durum notu
    if not detected_phone:
        cv2.putText(frame, "Telefon aranıyor...", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Buldum!", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Telefon Takip Sistemi v2.0", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break # 'q'ya basarsan dükkanı kapatırız

cap.release()
cv2.destroyAllWindows()