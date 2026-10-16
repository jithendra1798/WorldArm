"""Compatibility shim — perception lives in worldarm.perception."""
from worldarm.perception import capture, decode_color, decode_depth, deproject

__all__ = ["capture", "decode_color", "decode_depth", "deproject"]
