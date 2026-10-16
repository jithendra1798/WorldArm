"""Read and print the current joint positions of the configured Viam arm."""

import asyncio

from viam.components.arm import Arm

from connection import connect, env


async def main() -> None:
    async with await connect() as machine:
        arm = Arm.from_robot(machine, env("VIAM_ARM"))
        positions = await arm.get_joint_positions()

        for index, position in enumerate(positions.values, start=1):
            print(f"Joint {index}: {position}")


if __name__ == "__main__":
    asyncio.run(main())
