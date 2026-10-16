"""Read-only: track a black ball live on a webcam and show FPS. Moves nothing. Press q to quit."""

import time

import cv2
import numpy as np

CAMERA = 0                        # 0 = laptop webcam
LOW = np.array([0, 0, 0])         # black: any hue, any saturation...
HIGH = np.array([180, 255, 60])   # ...but dark (brightness under 60)
MIN_AREA = 200                    # ignore blobs smaller than this (pixels)
MIN_ROUNDNESS = 0.6               # 1.0 = perfect circle; cables and shadows score lower

cap = cv2.VideoCapture(CAMERA)
last = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("no camera frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOW, HIGH)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            break
        (x, y), r = cv2.minEnclosingCircle(c)
        roundness = area / (np.pi * r * r)
        if roundness > MIN_ROUNDNESS:
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 0, 255), 3)
            print(f"ball at ({x:.0f}, {y:.0f})  roundness {roundness:.2f}")
            break

    now = time.time()
    fps = 1 / (now - last)
    last = now
    cv2.putText(frame, f"{fps:.0f} fps", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("ball", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()