"""Read-only: track the black ball on webcam-1 through Viam and measure real FPS. Moves nothing."""

import asyncio
import time

import cv2
import numpy as np
from viam.components.camera import Camera

from worldarm.connection import connect
from worldarm.perception import decode_color

CAMERA_NAME = "overview-cam"
LOW = np.array([0, 0, 0])
HIGH = np.array([180, 255, 60])
MIN_AREA = 200
MIN_ROUNDNESS = 0.6


def find_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOW, HIGH)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            return None
        (x, y), r = cv2.minEnclosingCircle(c)
        if 8 < r < 30 and area / (np.pi * r * r) > MIN_ROUNDNESS:
            return x, y, r
    return None


async def main():
    machine = await connect()
    cam = Camera.from_robot(machine, CAMERA_NAME)
    t0 = time.time()
    for i in range(30):
        images, _ = await cam.get_images()
        frame = decode_color(images[0].data)
        ball = find_ball(frame)
        print(f"frame {i}: {frame.shape[1]}x{frame.shape[0]}  ball={ball}")
    print(f"\nreal speed: {30 / (time.time() - t0):.1f} fps")
    cv2.imwrite("captures/webcam1.jpg", frame)
    await machine.close()

asyncio.run(main())