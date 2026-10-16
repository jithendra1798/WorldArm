"""Compatibility shim — connection lives in worldarm.connection."""
from worldarm.connection import connect, env

__all__ = ["connect", "env"]
