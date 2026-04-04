from ultralytics import YOLO
import cv2
import pygame
import time
import threading
from datetime import datetime

# --- MODEL VE KAMERA AYARLARI ---
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Ses Sistemi Başlatma
pygame.mixer.init()
pygame.mixer.set_num_channels(8)

# --- DEĞİŞKENLER (EN AÇIK HALİYLE) ---
sol_ust_sayac = 0
sag_ust_sayac = 0
sol_alt_sayac = 0
sag_alt_sayac = 0

# Olay Geçmişi Listesi (Yeni olayları buraya tek tek ekleyeceğiz)
gecmis_listesi = [] 

# Zamanlama ve Takip Değişkenleri
su_anki_bolge = None
onceki_bolge = None
artirim_yapildi_mi = False
bolgeye_giris_zamani = 0
bekleme_suresi = 3.0 # 3 saniye kuralı

# Ses Dosyaları Haritası
sesler = {
    "sol_ust": "sounds/sol_ust.wav",
    "sag_ust": "sounds/sag_ust.wav",
    "sol_alt": "sounds/sol_alt.wav",
    "sag_alt": "sounds/sag_alt.wav"
}

# --- SES ÇALMA FONKSİYONU ---
def ses_cal(bolge_adi):
    try:
        dosya_yolu = sesler[bolge_adi]
        ses_objesi = pygame.mixer.Sound(dosya_yolu)
        ses_objesi.play()
    except:
        print("Ses dosyasi bulunamadi!")

# --- ANA DÖNGÜ ---
kare_sayaci = 0

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    yukseklik, genislik, _ = frame.shape
    orta_x = genislik // 2
    orta_y = yukseklik // 2

    # --- ŞEFFAF BEYAZ PANEL YAPIMI (ADIM ADIM) ---
    kopyalanan_kare = frame.copy()
    # Sağ tarafa beyaz bir kutu çiziyoruz
    cv2.rectangle(kopyalanan_kare, (genislik - 260, 0), (genislik, 300), (255, 255, 255), -1)
    # Bu kutuyu ana görüntüyle %50 oranında karıştırıyoruz (Şeffaflık)
    frame = cv2.addWeighted(kopyalanan_kare, 0.5, frame, 0.5, 0)

    # --- EKRAN ÇİZGİLERİ ---
    cv2.line(frame, (orta_x, 0), (orta_x, yukseklik), (0, 255, 0), 2)
    cv2.line(frame, (0, orta_y), (genislik, orta_y), (0, 255, 0), 2)

    # --- YOLO TESPİT ---
    if kare_sayaci % 2 == 0:
        sonuclar = model.predict(frame, conf=0.40, classes=[67], verbose=False)
    kare_sayaci = kare_sayaci + 1

    telefon_bulundu_mu = False
    su_anki_bolge = None

    # Eğer bir şey bulunduysa
    if sonuclar and len(sonuclar[0].boxes) > 0:
        kutu = sonuclar[0].boxes[0]
        koordinat = kutu.xyxy[0]
        x1 = int(koordinat[0])
        y1 = int(koordinat[1])
        x2 = int(koordinat[2])
        y2 = int(koordinat[3])
        
        merkez_x = (x1 + x2) // 2
        merkez_y = (y1 + y2) // 2
        telefon_bulundu_mu = True

        # Bölgeyi belirleyelim (Uzun yoldan)
        if merkez_x < orta_x and merkez_y < orta_y:
            su_anki_bolge = "sol_ust"
        elif merkez_x >= orta_x and merkez_y < orta_y:
            su_anki_bolge = "sag_ust"
        elif merkez_x < orta_x and merkez_y >= orta_y:
            su_anki_bolge = "sol_alt"
        else:
            su_anki_bolge = "sag_alt"

        # Kutuyu çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # --- 3 SANİYE MANTIĞI ---
        if su_anki_bolge != onceki_bolge:
            # Yeni bir yere geçti, kronometreyi sıfırla
            bolgeye_giris_zamani = time.time()
            onceki_bolge = su_anki_bolge
            artirim_yapildi_mi = False
        else:
            # Aynı yerdeyse geçen süreyi hesapla
            gecen_zaman = time.time() - bolgeye_giris_zamani
            
            if gecen_zaman >= bekleme_suresi:
                if artirim_yapildi_mi == False:
                    # Sayacı artır (Hangi bölgeyse ona ekle)
                    if su_anki_bolge == "sol_ust": sol_ust_sayac += 1
                    if su_anki_bolge == "sag_ust": sag_ust_sayac += 1
                    if su_anki_bolge == "sol_alt": sol_alt_sayac += 1
                    if su_anki_bolge == "sag_alt": sag_alt_sayac += 1
                    
                    # Log listesine ekle (Yeni geleni en başa eklemek için insert kullanıyoruz)
                    zaman_damgasi = datetime.now().strftime("%H:%M:%S")
                    yeni_olay = "[" + zaman_damgasi + "] " + su_anki_bolge.upper() + " +1"
                    gecmis_listesi.insert(0, yeni_olay)
                    
                    # Listeyi 8 tane ile sınırla (Fazlasını sil)
                    if len(gecmis_listesi) > 8:
                        gecmis_listesi.pop() # En sondakini çıkar
                    
                    # Sesi çal
                    threading.Thread(target=ses_cal, args=(su_anki_bolge,), daemon=True).start()
                    artirim_yapildi_mi = True
            else:
                # Geri sayımı göster
                kalan = bekleme_suresi - gecen_zaman
                cv2.putText(frame, "Bekle: " + str(round(kalan, 1)), (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    else:
        # Telefon yoksa her şeyi sıfırla
        onceki_bolge = None
        artirim_yapildi_mi = False

    # --- EKRAN YAZILARI (SOL TARAF) ---
    cv2.putText(frame, "sol_ust: " + str(sol_ust_sayac), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "sag_ust: " + str(sag_ust_sayac), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "sol_alt: " + str(sol_alt_sayac), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "sag_alt: " + str(sag_alt_sayac), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- GEÇMİŞ PANELİ (SAĞ TARAF - SİYAH METİN) ---
    cv2.putText(frame, "SON OLAYLAR", (genislik - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    y_yeri = 65
    for olay in gecmis_listesi:
        cv2.putText(frame, olay, (genislik - 250, y_yeri), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_yeri = y_yeri + 25

    # --- ALT DURUM MESAJI (TÜRKÇE KARAKTER HATASIZ) ---
    if telefon_bulundu_mu == False:
        cv2.putText(frame, "TELEFON BULUNAMADI", (20, yukseklik - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "TELEFON DURUMU: TAMAM", (20, yukseklik - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Telefon Takip Paneli", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()