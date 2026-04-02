from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

prev_state = "no_phone"
prev_region = "NONE"
phone_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, verbose=False)
    annotated_frame = frame.copy()
    h, w, _ = frame.shape

    phone_detected_this_frame = False
    region = "NONE"
    event_code = "none"
    event_text = ""

    cv2.line(annotated_frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    cv2.line(annotated_frame, (0, h // 2), (w, h // 2), (255, 255, 255), 2)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name == "cell phone":
            phone_detected_this_frame = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

            if center_x < w // 2 and center_y < h // 2:
                region = "TOP_LEFT"
            elif center_x >= w // 2 and center_y < h // 2:
                region = "TOP_RIGHT"
            elif center_x < w // 2 and center_y >= h // 2:
                region = "BOTTOM_LEFT"
            else:
                region = "BOTTOM_RIGHT"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                "cell phone",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    if phone_detected_this_frame:
        phone_frames += 1
    else:
        phone_frames = 0
        region = "NONE"

    if phone_frames > 5:
        state = "phone_detected"
    else:
        state = "no_phone"

    if state != prev_state:
        print("STATE CHANGED:", prev_state, "→", state)

    if region != prev_region and region != "NONE" and prev_region != "NONE":
        if prev_region == "TOP_RIGHT" and region == "TOP_LEFT":
            event_text = "telefon sag ustten sol uste gecti"
        elif prev_region == "TOP_LEFT" and region == "TOP_RIGHT":
            event_text = "telefon sol ustten sag uste gecti"
        elif prev_region == "BOTTOM_RIGHT" and region == "BOTTOM_LEFT":
            event_text = "telefon sag alttan sol alta gecti"
        elif prev_region == "BOTTOM_LEFT" and region == "BOTTOM_RIGHT":
            event_text = "telefon sol alttan sag alta gecti"

        if event_text != "":
            print("EVENT:", event_text)

    prev_state = state
    prev_region = region

    cv2.putText(annotated_frame, f"STATE: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(annotated_frame, f"REGION: {region}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(annotated_frame, f"EVENT: {event_text}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()