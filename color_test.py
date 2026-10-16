import cv2
import numpy as np

# HSV ranges for each block color (hue, saturation, brightness)
COLORS = {
    "yellow": ([20, 100, 100], [35, 255, 255]),
    "green":  ([40, 80, 50],   [85, 255, 255]),
    "blue":   ([95, 120, 50],  [130, 255, 255]),
}

img = cv2.imread("captures/color.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

for name, (low, high) in COLORS.items():
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        print(f"{name} at ({x},{y}) size {w}x{h}")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(img, name, (x, max(y - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imwrite("captures/all_colors.png", img)