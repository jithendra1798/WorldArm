import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient

# Each arm has its own env file; pick one with WORLDARM_ENV=.env.arm2 (default: .env). A file picked that way
# wins over variables already in the shell: VS Code's terminal pre-loads .env (python.terminal.useEnvFile),
# which otherwise kept pointing at the other arm.
load_dotenv(os.getenv("WORLDARM_ENV", ".env"), override="WORLDARM_ENV" in os.environ)


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
