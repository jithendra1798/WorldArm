import cv2
import numpy as np

img = cv2.imread("captures/color.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   # HSV makes color easier to pick out than RGB

# green range in HSV (hue, saturation, brightness)
low = np.array([40, 80, 50])
high = np.array([85, 255, 255])
mask = cv2.inRange(hsv, low, high)           # white = green pixels, black = everything else

print("green pixels:", cv2.countNonZero(mask))
cv2.imwrite("captures/green_mask.png", mask)
# find the green blobs and draw a box around each one
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    area = cv2.contourArea(c)
    if area < 500:            # ignore tiny specks of noise
        continue
    x, y, w, h = cv2.boundingRect(c)
    print(f"green blob at ({x},{y}) size {w}x{h}, area {area:.0f}")
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imwrite("captures/green_boxes.png", img)