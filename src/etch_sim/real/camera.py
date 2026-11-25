"""Scene-camera frame sources for RealBackend.

`RealSenseServer` is a client of `scripts/rs_server.py`, the process that holds the D435 for a whole
session (Linux, or Windows with `uv run python scripts/rs_server.py`). Each call asks the server for a
burst, reads the saved PNGs back and returns them as RGB arrays. The burst is what sketch_extract
expects: the median of the burst removes sensor noise.

The server listens on 127.0.0.1 only, so the drawing script and the server run on the same machine.
"""
from __future__ import annotations

import json
import socket
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

RS_HOST, RS_PORT = "127.0.0.1", 8766


class CameraServerError(RuntimeError):
    """The camera server is not running, or a grab failed."""


class RealSenseServer:
    def __init__(self, host: str = RS_HOST, port: int = RS_PORT, n: int = 10, timeout_s: float = 30.0,
                 save_dir: Path | None = None):
        """n: frames per burst. save_dir: keep every burst in save_dir/<time>_<count>/ (for checking an arm
        session afterwards); None = temporary files, deleted after reading."""
        self.addr, self.n, self.timeout_s = (host, port), n, timeout_s
        self.save_dir = None if save_dir is None else Path(save_dir)
        self.count = 0

    def call(self, req: dict) -> dict:
        try:
            with socket.create_connection(self.addr, timeout=self.timeout_s) as s, s.makefile("rw") as f:
                f.write(json.dumps(req) + "\n")
                f.flush()
                line = f.readline()
        except OSError as e:
            raise CameraServerError(f"camera server not reachable at {self.addr[0]}:{self.addr[1]} ({e}). "
                                    "Start it first: see README 'Real scene camera'.") from e
        if not line:
            raise CameraServerError("camera server closed the connection")
        return json.loads(line)

    def ping(self) -> bool:
        try:
            return bool(self.call({"cmd": "ping"}).get("ok"))
        except CameraServerError:
            return False

    def _grab_into(self, out: Path) -> list[np.ndarray]:
        r = self.call({"cmd": "grab", "out": str(out.resolve()), "n": self.n})
        files = sorted(out.glob("rgb_*.png"))
        if not r.get("ok") or not files:
            raise CameraServerError(f"grab failed: {r.get('error') or 'no frames saved'}")
        return [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]

    def __call__(self) -> list[np.ndarray]:
        self.count += 1
        if self.save_dir is not None:
            out = self.save_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{self.count:04d}"
            return self._grab_into(out)
        with tempfile.TemporaryDirectory(prefix="etch_rs_") as d:
            return self._grab_into(Path(d))
