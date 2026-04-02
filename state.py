from ultralytics import YOLO
import cv2
from collections import defaultdict

# YOLO modelini yükle
model = YOLO("yolov8n.pt")

# Kamera aç
cap = cv2.VideoCapture(0)

# State tutma
zone_counts = defaultdict(int)
last_zone = None
last_center = None

# COCO class id: cell phone = 67
PHONE_CLASS_ID = 67

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

    results = model(frame)

    detected_phone = False
    current_zone = None

    for box in results[0].boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        # Sadece telefonu al
        if cls == PHONE_CLASS_ID and conf > 0.40:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            current_zone = get_zone(cx, cy, w, h)
            detected_phone = True

            # Kutu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Bilgi yaz
            cv2.putText(
                frame,
                f"Telefon | {current_zone} | conf:{conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # Aynı frame'de ilk telefon yeter
            break

    if detected_phone:
        # İlk kez görüldüyse veya zone değiştiyse event üret
        if current_zone != last_zone:
            zone_counts[current_zone] += 1

            if last_zone is None:
                print(f"EVENT: telefon ilk kez {current_zone} bölgesinde goruldu")
            else:
                print(f"EVENT: telefon {last_zone} -> {current_zone} gecti")

            print(f"STATE: telefon {zone_counts[current_zone]} defa {current_zone} bölgede goruldu")

            last_zone = current_zone

        # Ekrana state yaz
        y_offset = 60
        for zone_name in ["sol_ust", "sag_ust", "sol_alt", "sag_alt"]:
            text = f"{zone_name}: {zone_counts[zone_name]}"
            cv2.putText(
                frame,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )
            y_offset += 30

        cv2.putText(
            frame,
            f"Anlik Zone: {last_zone}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:
        cv2.putText(
            frame,
            "Telefon tespit edilmedi",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("PlayX Zone + Event + State", frame)

    # q ile çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()