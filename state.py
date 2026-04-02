from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

prev_state = "no_phone"
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

    if phone_frames > 5:
        state = "phone_detected"
    else:
        state = "no_phone"

    if state != prev_state:
        print("STATE CHANGED:", prev_state, "→", state)

    prev_state = state

    cv2.putText(
        annotated_frame,
        f"STATE: {state}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"PHONE FRAMES: {phone_frames}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"REGION: {region}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("YOLO Cell Phone Filter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()