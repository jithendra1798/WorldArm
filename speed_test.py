"""Read-only: time each step of the tracker. Moves nothing."""
import asyncio, time
import matplotlib.pyplot as plt
from viam.components.camera import Camera
from worldarm.connection import connect
from worldarm.perception import decode_color
from track_throw import find_ball

async def main():
    machine = await connect()
    cam = Camera.from_robot(machine, "overview-cam")
    get_t, dec_t, find_t = [], [], []
    for _ in range(20):
        a = time.time(); images, _ = await cam.get_images(filter_source_names=["color"])
        b = time.time(); frame = decode_color(images[0].data)
        c = time.time(); find_ball(frame)
        d = time.time()
        get_t.append((b - a) * 1000); dec_t.append((c - b) * 1000); find_t.append((d - c) * 1000)
    await machine.close()

    avg = lambda x: sum(x) / len(x)
    names = ["get picture", "decode", "find ball"]
    vals = [avg(get_t), avg(dec_t), avg(find_t)]
    for n, v in zip(names, vals):
        print(f"{n:12s} {v:6.1f} ms")
    print(f"total        {sum(vals):6.1f} ms  ->  {1000 / sum(vals):.1f} fps")

    plt.bar(names, vals); plt.ylabel("ms per frame"); plt.title("Where the time goes")
    plt.savefig("captures/speed.png")
    print("chart saved: captures/speed.png")

if __name__ == "__main__":
    asyncio.run(main())