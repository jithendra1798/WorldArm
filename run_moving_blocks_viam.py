"""Trigger the on-robot moving-blocks Generic service (PickBottle-style).

The transfer logic runs on armfarm05 inside Viam — this script only connects
and calls DoCommand, same pattern as armfarm03's PickBottle client.

Preview (no motion):
  python run_moving_blocks_viam.py

Execute transfer (requires enable_motion=true on the service):
  python run_moving_blocks_viam.py --move
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from viam.services.generic import Generic as GenericService

from connection import connect


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run moving-blocks via Viam DoCommand")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Execute motion (requires enable_motion=true on the service)",
    )
    parser.add_argument(
        "--service",
        default=os.getenv("VIAM_MOVING_BLOCKS_SERVICE", "moving-blocks"),
        help="Generic service name on the machine",
    )
    args = parser.parse_args()

    async with await connect() as machine:
        print("Resources:")
        print(machine.resource_names)
        service = GenericService.from_robot(machine, args.service)
        if args.move:
            command = {"action": "move", "confirm": True}
            print(f"Calling {args.service} DoCommand move (hardware will move)...")
        else:
            command = {"action": "preview"}
            print(f"Calling {args.service} DoCommand preview (no motion)...")
        result = await service.do_command(command)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
