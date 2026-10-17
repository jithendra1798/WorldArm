"""Two Viam machines at once, each from its own env file: armfarm05 throws, armfarm03 watches and catches.

dotenv_values reads a file without touching os.environ, so the two machines' keys never mix (the chapter 1
connection module loads one file into the environment, which can't hold two).
"""

from __future__ import annotations

from dotenv import dotenv_values
from viam.robot.client import RobotClient

NEEDED = ("VIAM_API_KEY", "VIAM_API_KEY_ID", "VIAM_MACHINE_ADDRESS")


def machine_env(path: str) -> dict:
    """The settings in one env file (empty values left out)."""
    vals = {k: v for k, v in dotenv_values(path).items() if v}
    missing = [k for k in NEEDED if k not in vals]
    if missing:
        raise SystemExit(f"{path}: {', '.join(missing)} not set (Viam app -> the machine -> CONNECT -> API keys)")
    return vals


def check_machine(vals: dict, name: str, path: str) -> None:
    """Refuse an env file that points at another machine (a swapped file would move the wrong arm)."""
    host = vals["VIAM_MACHINE_ADDRESS"].split(".")[0]
    if name.replace("0", "") not in host.replace("0", ""):
        raise SystemExit(f"{path} points at {host}, not {name}")


def component(vals: dict, key: str, default: str) -> str:
    return vals.get(key, default)


async def connect(path: str, name: str) -> tuple[RobotClient, dict]:
    vals = machine_env(path)
    check_machine(vals, name, path)
    opts = RobotClient.Options.with_api_key(api_key=vals["VIAM_API_KEY"], api_key_id=vals["VIAM_API_KEY_ID"])
    return await RobotClient.at_address(vals["VIAM_MACHINE_ADDRESS"], opts), vals
