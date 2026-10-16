"""Call the on-robot pick-red-cube generic service (preview by default).

Does not implement perception locally — the Viam module on the machine does.
Default action is preview (no motion). Pick requires the service config
``enable_pick: true`` and ``--pick``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from viam.services.generic import Generic

from connection import connect


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service",
        default=os.getenv("VIAM_PICK_SERVICE", "pick-red-cube"),
        help="Generic service name on the machine",
    )
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Send action=pick (refused by the module unless enable_pick is true)",
    )
    args = parser.parse_args()
    action = "pick" if args.pick else "preview"

    async with await connect() as machine:
        svc = Generic.from_robot(machine, args.service)
        result = await svc.do_command({"action": action})
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
