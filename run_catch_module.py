"""Call the on-robot catch-thrown-object generic service (preview by default)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.services.generic import Generic


async def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service",
        default=os.getenv("VIAM_CATCH_SERVICE", "catch-thrown-object"),
    )
    parser.add_argument("--ready", action="store_true", help="arm_ready (needs enable_catch)")
    parser.add_argument(
        "--catch",
        action="store_true",
        help="catch attempt (needs enable_catch + safety ack)",
    )
    parser.add_argument("--abort", action="store_true")
    args = parser.parse_args()

    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    machine = await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)
    try:
        svc = Generic.from_robot(machine, args.service)
        if args.abort:
            cmd = {"action": "abort"}
        elif args.catch:
            cmd = {
                "action": "catch",
                "safety_ack": "I_UNDERSTAND_CATCH_RISKS",
            }
        elif args.ready:
            cmd = {"action": "arm_ready"}
        else:
            cmd = {"action": "preview"}
        result = await svc.do_command(cmd)
        print(json.dumps(result, indent=2, default=str))
    finally:
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
