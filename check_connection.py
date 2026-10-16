"""Connect to the machine and read the arm's current joint positions.

This script only reads state; it does not move the arm.
"""

import asyncio

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper

from connection import connect, env


async def main():
    async with await connect() as machine:
        print("Resources:")
        for name in machine.resource_names:
            print(f"  {name.subtype:<12} {name.name}")

        arm = Arm.from_robot(machine, env("VIAM_ARM"))
        print(f"\nArm end pose: {await arm.get_end_position()}")
        print(f"Arm joint positions (degrees): {await arm.get_joint_positions()}")

        cam = Camera.from_robot(machine, env("VIAM_CAMERA"))
        images, _ = await cam.get_images()
        for img in images:
            print(f"Camera image '{img.name}': {img.mime_type}, {len(img.data)} bytes")

        gripper = Gripper.from_robot(machine, env("VIAM_GRIPPER"))
        print(f"Gripper moving: {await gripper.is_moving()}")


if __name__ == "__main__":
    asyncio.run(main())
