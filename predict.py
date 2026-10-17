"""Landing predictor tested on fake throws. No robot, no camera."""
import matplotlib.pyplot as plt
import numpy as np

FPS = 23            # your real camera speed
FLIGHT_S = 0.5      # how long the ball is in the air
NOISE_MM = 5        # how much the tracker wobbles
rng = np.random.default_rng(0)

def fake_throw():
    start = rng.uniform(-50, 50, 2)                  # where it's released (mm)
    speed = rng.uniform(300, 900)                    # mm per second (safety cap 900)
    angle = rng.uniform(-0.3, 0.3)                   # direction (radians)
    vel = speed * np.array([np.cos(angle), np.sin(angle)])
    t = np.arange(0, FLIGHT_S, 1 / FPS)
    seen = start + np.outer(t, vel) + rng.normal(0, NOISE_MM, (len(t), 2))
    landing = start + vel * FLIGHT_S
    return t, seen, landing

def predict(t, seen, t_land):
    """Fit a straight line to what we saw, extend it to landing time."""
    fx = np.polyfit(t, seen[:, 0], 1)
    fy = np.polyfit(t, seen[:, 1], 1)
    return np.array([np.polyval(fx, t_land), np.polyval(fy, t_land)])

frames_seen = range(2, 11)
avg_err = []
for n in frames_seen:
    errs = []
    for _ in range(200):
        t, seen, landing = fake_throw()
        guess = predict(t[:n], seen[:n], FLIGHT_S)
        errs.append(np.linalg.norm(guess - landing))
    avg_err.append(np.mean(errs))
    print(f"saw {n:2d} frames ({n / FPS * 1000:3.0f} ms): off by {np.mean(errs):5.1f} mm")

plt.plot(list(frames_seen), avg_err, "o-")
plt.axhline(30, linestyle="--", color="gray", label="cup radius ~30 mm")
plt.xlabel("frames seen before guessing"); plt.ylabel("landing error (mm)")
plt.title("How many frames until the guess is good enough?"); plt.legend()
plt.savefig("captures/predict.png")
print("chart saved: captures/predict.png")