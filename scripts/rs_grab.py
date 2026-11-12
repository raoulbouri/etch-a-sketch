"""Minimal RealSense grab: 10 colour frames at 1280x720, 6 fps (a mode the D435 offers on USB 2).

    sh scripts/rs_capture.sh grab [output_dir]      (Linux)

Saves rgb_00.png ... rgb_09.png in output_dir (default test_images/rs/<time>/) for scripts/analyze_sketch.py.
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

out = (Path(sys.argv[1]) if len(sys.argv) > 1
       else Path(__file__).resolve().parents[1] / "test_images" / "rs" / time.strftime("%Y%m%d-%H%M%S"))
pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
pipe.start(cfg)
if os.environ.get("RS_READY_FILE"):
    Path(os.environ["RS_READY_FILE"]).touch()   # camera is ours: the wrapper stops killing camera services
try:
    for _ in range(12):                      # ~2 s for auto exposure to settle
        pipe.wait_for_frames(3000)
    frames = [np.asanyarray(pipe.wait_for_frames(3000).get_color_frame().get_data()).copy() for _ in range(10)]
finally:
    pipe.stop()
out.mkdir(parents=True, exist_ok=True)
for i, f in enumerate(frames):
    cv2.imwrite(str(out / f"rgb_{i:02d}.png"), f)
print(f"saved {len(frames)} frames to {out}")
