"""Capture still frames from the Intel RealSense D435 scene camera. Read-only: no arm.

Default: colour (RGB) only. With --all: colour, depth aligned to the colour image, and left infrared.

The IR dot projector is switched ON only while depth frames are taken and OFF for colour and infrared,
so the dots do not cover the screen. Every frame of a burst is saved, plus a per-pixel median (noise)
and minimum (keeps a thin dark line that flickers) of the burst.

Needs pyrealsense2 (Linux: `uv sync --group realsense`; blocked on macOS). Run it through the wrapper:

    sh scripts/rs_capture.sh                    # RGB burst
    sh scripts/rs_capture.sh --all              # RGB + depth + IR
    sh scripts/rs_capture.sh --all --exposure 120 --frames 12 --tag square
    sh scripts/rs_capture.sh --max-width 1280      # cap the colour size

Output: test_images/rs/<YYYYmmdd-HHMMSS>[_tag]/ (the folder is git-ignored; clean it whenever you like).
Only numpy, OpenCV and pyrealsense2 are imported, so any Python that has those works.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parents[1] / "test_images" / "rs"


def intr_dict(profile) -> dict:
    i = profile.as_video_stream_profile().get_intrinsics()
    return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
            "model": str(i.model), "coeffs": list(i.coeffs)}


def device_modes(dev) -> dict:
    """{(stream, index, format): set of (w, h, fps)} that this device offers on this USB link."""
    modes: dict = {}
    for sensor in dev.query_sensors():
        for p in sensor.get_stream_profiles():
            if p.is_video_stream_profile():
                v = p.as_video_stream_profile()
                modes.setdefault((p.stream_type(), p.stream_index(), p.format()), set()).add((v.width(), v.height(), p.fps()))
    return modes


def candidates(modes: dict, args, with_depth: bool):
    """Stream combinations to try, best first: largest colour, then largest depth/IR, same fps for all.

    Only modes the device lists are tried, and each is checked with can_resolve before start, so a
    USB 2 link (fewer, slower modes) is handled instead of failing on a fixed 1080p/720p request.
    """
    color = sorted((m for m in modes.get((rs.stream.color, 0, rs.format.bgr8), ())
                    if m[0] <= args.max_width and m[2] <= args.max_fps), key=lambda m: (-m[0] * m[1], m[2]))
    depth = sorted(modes.get((rs.stream.depth, 0, rs.format.z16), ()), key=lambda m: (-m[0] * m[1], m[2]))
    ir = modes.get((rs.stream.infrared, 1, rs.format.y8), set())
    for cw, ch, fps in color:
        if not with_depth:
            yield (cw, ch, fps), None
            continue
        for dw, dh, dfps in depth:
            if dfps == fps and (dw, dh, fps) in ir:
                yield (cw, ch, fps), (dw, dh)


def start(pipe: rs.pipeline, dev, args, with_depth: bool):
    modes = device_modes(dev)
    tried = 0
    for (cw, ch, fps), d in candidates(modes, args, with_depth):
        cfg = rs.config()
        cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, fps)
        if d:
            cfg.enable_stream(rs.stream.depth, d[0], d[1], rs.format.z16, fps)
            cfg.enable_stream(rs.stream.infrared, 1, d[0], d[1], rs.format.y8, fps)
        tried += 1
        if not cfg.can_resolve(rs.pipeline_wrapper(pipe)):
            continue
        desc = f"colour {cw}x{ch}" + (f", depth+IR {d[0]}x{d[1]}" if d else "") + f" @ {fps} fps"
        try:
            prof = pipe.start(cfg)
        except RuntimeError as e:
            print(f"{desc} refused at start: {e}")
            continue
        print("streaming", desc)
        return prof, desc, fps
    listing = {f"{st} {fmt}": sorted(m) for (st, idx, fmt), m in modes.items() if st == rs.stream.color or idx <= 1}
    raise SystemExit(f"no usable stream combination ({tried} tried). Device modes on this link:\n"
                     + json.dumps({k: [list(x) for x in v] for k, v in listing.items()}, indent=1))


def set_opt(sensor, opt, value) -> bool:
    if sensor.supports(opt):
        sensor.set_option(opt, value)
        return True
    return False


def grab(pipe, n: int, settle: int, align=None):
    for _ in range(settle):
        pipe.wait_for_frames(3000)                       # raises after 3 s instead of blocking forever
    out = []
    for _ in range(n):
        fs = pipe.wait_for_frames(3000)
        out.append(align.process(fs) if align else fs)
    return out


def burst_stats(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    st = np.stack(frames)
    return np.median(st, axis=0).astype(st.dtype), st.min(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also depth (aligned to colour) and infrared")
    ap.add_argument("--frames", type=int, default=8, help="frames per burst")
    ap.add_argument("--max-width", type=int, default=1920, help="largest colour width to use")
    ap.add_argument("--max-fps", type=int, default=30)
    ap.add_argument("--exposure", type=float, help="manual colour exposure (units of 0.1 ms); default auto")
    ap.add_argument("--gain", type=float, help="manual colour gain (with --exposure)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    ctx = rs.context()
    if not len(ctx.query_devices()):
        raise SystemExit("no RealSense found (plugged in? on Linux check the udev rules, see README)")
    dev = ctx.query_devices()[0]
    info = {k: dev.get_info(getattr(rs.camera_info, k)) for k in
            ("name", "serial_number", "firmware_version", "usb_type_descriptor")
            if dev.supports(getattr(rs.camera_info, k))}
    print("device:", info)
    if info.get("usb_type_descriptor", "3").startswith("2"):
        print("WARNING: camera is on a USB 2 link: fewer, slower modes, and macOS hides its colour webcam."
              " Use a USB 3 cable and port (the blue-tongue USB-A end or a USB-C 3.x cable).")

    pipe = rs.pipeline(ctx)
    profile, desc, fps = start(pipe, dev, a, a.all)
    if os.environ.get("RS_READY_FILE"):
        Path(os.environ["RS_READY_FILE"]).touch()        # camera is ours: the wrapper stops killing camera services
    settle = max(5, 2 * fps)                             # ~2 s for auto exposure, whatever the frame rate
    out = ROOT / (time.strftime("%Y%m%d-%H%M%S") + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=True)               # only once streaming works: no empty folders
    meta = {"device": info, "mode": "all" if a.all else "rgb", "streams": desc, "frames": a.frames, "fps": fps,
            "color": intr_dict(profile.get_stream(rs.stream.color))}
    try:
        color_sensor = profile.get_device().first_color_sensor()
        if a.exposure is not None:
            set_opt(color_sensor, rs.option.enable_auto_exposure, 0)
            set_opt(color_sensor, rs.option.exposure, a.exposure)
            if a.gain is not None:
                set_opt(color_sensor, rs.option.gain, a.gain)
        else:
            set_opt(color_sensor, rs.option.enable_auto_exposure, 1)

        if a.all:
            depth_sensor = profile.get_device().first_depth_sensor()
            meta["depth_scale_m"] = depth_sensor.get_depth_scale()
            meta["infrared"] = intr_dict(profile.get_stream(rs.stream.infrared, 1))
            # 1) depth with the projector on, aligned to the colour image
            set_opt(depth_sensor, rs.option.emitter_enabled, 1)
            fsets = grab(pipe, a.frames, settle=settle, align=rs.align(rs.stream.color))
            d = np.stack([np.asanyarray(f.get_depth_frame().get_data()) for f in fsets]).astype(np.float32)
            d[d == 0] = np.nan                                   # 0 = no depth there
            with np.errstate(all="ignore"):
                dmed = np.nanmedian(d, axis=0)
            depth_mm = np.nan_to_num(dmed * meta["depth_scale_m"] * 1000.0, nan=0).astype(np.uint16)
            np.save(out / "depth_mm_aligned.npy", depth_mm)
            valid = depth_mm[depth_mm > 0]
            if valid.size:
                lo, hi = np.percentile(valid, [2, 98])
                vis = np.clip((depth_mm.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
                vis[depth_mm == 0] = 0
                cv2.imwrite(str(out / "depth_vis.png"), vis)
                meta["depth_valid_frac"] = float((depth_mm > 0).mean())
                meta["depth_median_mm"] = float(np.median(valid))
            # 2) projector off for colour and infrared
            set_opt(depth_sensor, rs.option.emitter_enabled, 0)

        fsets = grab(pipe, a.frames, settle=settle)
        rgb = [np.asanyarray(f.get_color_frame().get_data()).copy() for f in fsets]
        for i, im in enumerate(rgb):
            cv2.imwrite(str(out / f"rgb_{i:02d}.png"), im)
        med, mn = burst_stats(rgb)
        cv2.imwrite(str(out / "rgb_median.png"), med)
        cv2.imwrite(str(out / "rgb_min.png"), mn)
        cf = fsets[-1].get_color_frame()
        meta["color_exposure"] = (cf.get_frame_metadata(rs.frame_metadata_value.actual_exposure)
                                  if cf.supports_frame_metadata(rs.frame_metadata_value.actual_exposure) else None)
        meta["color_saturated_frac"] = float((med.max(axis=2) >= 250).mean())

        if a.all:
            ir = [np.asanyarray(f.get_infrared_frame(1).get_data()).copy() for f in fsets]
            for i, im in enumerate(ir):
                cv2.imwrite(str(out / f"ir_{i:02d}.png"), im)
            med, mn = burst_stats(ir)
            cv2.imwrite(str(out / "ir_median.png"), med)
            cv2.imwrite(str(out / "ir_min.png"), mn)
    finally:
        pipe.stop()
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{desc}; saved to {out}")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("color", "infrared")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
