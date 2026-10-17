"""Read-only: record the ball for 5 s during a throw. Saves CSV + chart. Moves nothing."""
import asyncio, csv, time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from viam.components.camera import Camera
from worldarm.connection import connect
from worldarm.perception import decode_color

CAMERA_NAME = "overview-cam"
LOW = np.array([0, 0, 0]); HIGH = np.array([180, 255, 60])
MIN_AREA = 200; MIN_ROUNDNESS = 0.6
TOP = 150  # rows above this are ignored (stops the fake hits we had before)

def find_ball(frame):
    part = frame[TOP:560, :]
    hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOW, HIGH)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            return None
        (x, y), r = cv2.minEnclosingCircle(c)
        if 8 < r < 30 and area / (np.pi * r * r) > MIN_ROUNDNESS:
            return x, y + TOP, r
    return None

async def main():
    machine = await connect()
    cam = Camera.from_robot(machine, CAMERA_NAME)
    print("Recording in 3 s... throw when it says GO")
    await asyncio.sleep(3)
    print("GO")
    rows, t0 = [], time.time()
    while time.time() - t0 < 5:
        images, _ = await cam.get_images(filter_source_names=["color"])
        frame = decode_color(images[0].data)
        ball = find_ball(frame)
        t = time.time() - t0
        rows.append((t, *(ball if ball else (None, None, None))))
    await machine.close()

    with open("captures/throw.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "x", "y", "r"]); w.writerows(rows)

    found = [r for r in rows if r[1] is not None]
    print(f"{len(rows)} frames, ball found in {len(found)}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([r[1] for r in found], [r[2] for r in found], "o-")
    ax[0].invert_yaxis(); ax[0].set_title("Ball path in the picture"); ax[0].set_xlabel("x px"); ax[0].set_ylabel("y px")
    ax[1].plot([r[0] for r in found], [r[1] for r in found], "o-", label="x")
    ax[1].plot([r[0] for r in found], [r[2] for r in found], "o-", label="y")
    ax[1].set_title("Position over time"); ax[1].set_xlabel("seconds"); ax[1].legend()
    plt.tight_layout(); plt.savefig("captures/throw.png")
    print("chart saved: captures/throw.png")

if __name__ == "__main__":
    asyncio.run(main())