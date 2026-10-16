"""Episode logs: runs/<timestamp>-<tag>/{meta.json, episode.jsonl, summary.json}.

meta.json records what's needed to reproduce a run (args, config, git commit, dirty flag).
"""

from __future__ import annotations

import dataclasses
import enum
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if hasattr(obj, "to_json"):
        return obj.to_json()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Message):
        return MessageToDict(obj)
    if hasattr(obj, "tolist"):  # numpy
        return obj.tolist()
    return str(obj)


def git_info() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


class EpisodeLogger:
    def __init__(self, root: Path | str, tag: str, meta: dict):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        root = Path(root)
        self.dir = root / f"{stamp}-{tag}"
        n = 1
        while self.dir.exists():
            n += 1
            self.dir = root / f"{stamp}-{tag}-{n}"
        self.dir.mkdir(parents=True)
        meta = {**meta, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_info()}
        self._write("meta.json", meta)
        self._steps = open(self.dir / "episode.jsonl", "a")

    def _write(self, name: str, data: dict) -> None:
        (self.dir / name).write_text(json.dumps(data, indent=2, default=to_jsonable) + "\n")

    def step(self, record: dict) -> None:
        self._steps.write(json.dumps(record, default=to_jsonable) + "\n")
        self._steps.flush()

    def finish(self, summary: dict) -> None:
        self._steps.close()
        self._write("summary.json", summary)
