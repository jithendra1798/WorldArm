"""Read-only: show what pictures overview-cam sends and how big they are."""
import asyncio
from viam.components.camera import Camera
from worldarm.connection import connect

async def main():
    machine = await connect()
    cam = Camera.from_robot(machine, "overview-cam")
    images, _ = await cam.get_images()
    for im in images:
        print(f"name={im.name}  type={im.mime_type}  size={len(im.data) / 1000:.0f} KB")
    await machine.close()

asyncio.run(main())