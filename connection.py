import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient

load_dotenv()


def env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set; add it to .env")
    return value


async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=env("VIAM_API_KEY"),
        api_key_id=env("VIAM_API_KEY_ID"),
    )
    return await RobotClient.at_address(env("VIAM_MACHINE_ADDRESS"), opts)
