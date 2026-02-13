#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera benchmark for Raspberry Pi (CSI/IMX477 and others)

Backends tested:
  1) Picamera2 (libcamera)
  2) OpenCV V4L2 (/dev/video*)
  3) OpenCV GStreamer (libcamerasrc) - if available

Outputs:
  - ./cam_benchmark_YYYYmmdd_HHMMSS/report.txt
  - sample images for each successful mode

Run:
  python3 cam_benchmark.py
Optional:
  python3 cam_benchmark.py --show   (try to show preview windows)
  python3 cam_benchmark.py --frames 120  (frames per test)
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ------------------------------ config ------------------------------

DEFAULT_RESOLUTIONS = [
    (4056, 3040),
    (2028, 1520),
    (1920, 1080),
    (1332, 990),
    (1280, 720),
    (640, 480),
]

# Реалистичные кандидаты FPS. Не все камеры/режимы поддержат — это нормально.
DEFAULT_FPS = [5, 10, 15, 24, 30, 50, 60, 90, 120]

# Сколько кадров брать для замера FPS
DEFAULT_FRAMES_PER_TEST = 90

# Сколько кадров "прогреть" перед замером
WARMUP_FRAMES = 10


# ------------------------------ helpers ------------------------------

def have_display() -> bool:
    return bool(os.environ.get("DISPLAY")) or sys.platform.startswith("win") or sys.platform == "darwin"


def run_cmd(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"[cmd failed] {cmd}: {e}"


def opencv_has_gstreamer() -> bool:
    try:
        info = cv2.getBuildInformation()
        # "GStreamer: YES" appears in build info
        return "GStreamer" in info and ("YES" in info.split("GStreamer:")[-1].splitlines()[0])
    except Exception:
        return False


def safe_imwrite(path: Path, bgr: np.ndarray) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), bgr)
        return bool(ok)
    except Exception:
        return False


def to_bgr(frame: np.ndarray) -> np.ndarray:
    # Picamera2 может дать RGB; OpenCV обычно BGR.
    if frame is None:
        return frame
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        # Тут не угадаешь 100% (RGB/BGR), но для сохранения сойдёт.
        # Для Picamera2 ниже мы явно конвертируем RGB->BGR.
        return frame
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def measure_fps_grab(get_frame_fn, frames_to_measure: int, warmup: int = WARMUP_FRAMES) -> Tuple[float, Optional[np.ndarray], str]:
    """
    Возвращает:
      fps_measured,
      last_frame (как numpy),
      error_text ("" если ок)
    """
    try:
        # warmup
        last = None
        for _ in range(warmup):
            last = get_frame_fn()
            if last is None:
                return 0.0, None, "warmup: got None frame"

        t0 = time.perf_counter()
        ok_frames = 0
        for _ in range(frames_to_measure):
            fr = get_frame_fn()
            if fr is None:
                return 0.0, last, f"measure: got None at frame {ok_frames}"
            last = fr
            ok_frames += 1
        t1 = time.perf_counter()

        dt = max(t1 - t0, 1e-9)
        fps = ok_frames / dt
        return fps, last, ""
    except Exception as e:
        return 0.0, None, f"exception: {type(e).__name__}: {e}"


@dataclass
class TestResult:
    backend: str
    device: str
    resolution: Tuple[int, int]
    fps_req: int
    opened: bool
    configured: bool
    fps_measured: float
    frame_shape: str
    frame_dtype: str
    sample_path: str
    error: str


# ------------------------------ Picamera2 backend ------------------------------

class Picamera2Bench:
    def __init__(self):
        self.available = False
        self.picamera2 = None
        self.Picamera2 = None
        self.cam = None
        self.cam_id = None

        try:
            import importlib
            if importlib.util.find_spec("picamera2") is None:
                return
            self.picamera2 = importlib.import_module("picamera2")
            self.Picamera2 = self.picamera2.Picamera2
            self.available = True
        except Exception:
            self.available = False

    def list_cameras(self) -> List[str]:
        if not self.available:
            return []
        try:
            infos = self.Picamera2.global_camera_info()
            if not infos:
                return ["0"]
            return [str(i) for i in range(len(infos))]
        except Exception:
            return ["0"]

    def open(self, cam_id: str) -> Tuple[bool, str]:
        if not self.available:
            return False, "picamera2 not available"
        try:
            if self.cam is not None and self.cam_id != cam_id:
                try:
                    self.cam.stop()
                except Exception:
                    pass
                try:
                    self.cam.close()
                except Exception:
                    pass
                self.cam = None

            if self.cam is None:
                self.cam = self.Picamera2(int(cam_id))
                self.cam_id = cam_id
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def configure(self, size: Tuple[int, int], fps: int) -> Tuple[bool, str]:
        """
        Стараемся настроить preview поток с RGB888 (самый удобный для OpenCV),
        задаём fps через FrameDurationLimits (микросекунды).
        """
        if self.cam is None:
            return False, "camera not opened"
        try:
            # frame duration in us
            if fps > 0:
                us = int(1_000_000 / fps)
                controls = {"FrameDurationLimits": (us, us)}
            else:
                controls = {}

            cfg = self.cam.create_preview_configuration(main={"size": size, "format": "RGB888"})
            self.cam.configure(cfg)
            if controls:
                self.cam.set_controls(controls)
            self.cam.start()
            time.sleep(0.25)
            return True, ""
        except Exception as e:
            try:
                self.cam.stop()
            except Exception:
                pass
            return False, f"{type(e).__name__}: {e}"

    def get_frame(self) -> Optional[np.ndarray]:
        if self.cam is None:
            return None
        arr = self.cam.capture_array()
        if arr is None:
            return None
        # arr приходит как RGB888 -> конвертим в BGR для OpenCV
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    def close(self):
        if self.cam is not None:
            try:
                self.cam.stop()
            except Exception:
                pass
            try:
                self.cam.close()
            except Exception:
                pass
            self.cam = None
            self.cam_id = None


# ------------------------------ OpenCV V4L2 backend ------------------------------

class OpenCVV4L2Bench:
    def __init__(self):
        pass

    def list_devices(self, max_index: int = 10) -> List[str]:
        devs = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                devs.append(str(idx))
            cap.release()
        return devs

    def run_test(self, dev_id: str, size: Tuple[int, int], fps: int, frames_to_measure: int) -> Tuple[bool, bool, float, Optional[np.ndarray], str]:
        cap = cv2.VideoCapture(int(dev_id), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False, False, 0.0, None, "VideoCapture not opened"

        # Configure
        w, h = size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        if fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(fps))

        def get_frame():
            ok, fr = cap.read()
            return fr if ok else None

        fps_meas, last, err = measure_fps_grab(get_frame, frames_to_measure)
        cap.release()
        return True, True if err == "" else True, fps_meas, last, err


# ------------------------------ OpenCV GStreamer backend ------------------------------

class OpenCVGstBench:
    def __init__(self):
        self.enabled = opencv_has_gstreamer()

    def pipeline(self, size: Tuple[int, int], fps: int) -> str:
        w, h = size
        # libcamerasrc -> appsink (BGR)
        # NOTE: framerate needs like "30/1"
        fps_str = f"{int(fps)}/1" if fps > 0 else "30/1"
        return (
            "libcamerasrc ! "
            f"video/x-raw,width={w},height={h},framerate={fps_str} ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false"
        )

    def run_test(self, size: Tuple[int, int], fps: int, frames_to_measure: int) -> Tuple[bool, bool, float, Optional[np.ndarray], str]:
        if not self.enabled:
            return False, False, 0.0, None, "OpenCV built without GStreamer"

        pipe = self.pipeline(size, fps)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            return False, False, 0.0, None, "GStreamer VideoCapture not opened"

        def get_frame():
            ok, fr = cap.read()
            return fr if ok else None

        fps_meas, last, err = measure_fps_grab(get_frame, frames_to_measure)
        cap.release()
        return True, True if err == "" else True, fps_meas, last, err


# ------------------------------ main benchmark ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="try to show images in windows")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES_PER_TEST, help="frames per test")
    ap.add_argument("--max-v4l2", type=int, default=10, help="max /dev/video index to probe via OpenCV")
    args = ap.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"cam_benchmark_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.txt"

    results: List[TestResult] = []

    # System info
    sys_info = []
    sys_info.append(f"Time: {timestamp}")
    sys_info.append(f"Platform: {platform.platform()}")
    sys_info.append(f"Python: {sys.version.replace(os.linesep, ' ')}")
    sys_info.append(f"OpenCV: {cv2.__version__}")
    sys_info.append(f"OpenCV GStreamer: {'YES' if opencv_has_gstreamer() else 'NO'}")
    sys_info.append(f"uname -a: {run_cmd(['uname', '-a'])}")
    sys_info.append(f"libcamera-hello --version: {run_cmd(['bash', '-lc', 'libcamera-hello --version 2>/dev/null || true'])}")
    sys_info.append("")

    # Backends init
    picam = Picamera2Bench()
    v4l2 = OpenCVV4L2Bench()
    gst = OpenCVGstBench()

    # Candidate lists
    resolutions = DEFAULT_RESOLUTIONS
    fps_list = DEFAULT_FPS

    # -------- Picamera2 tests --------
    if picam.available:
        cam_ids = picam.list_cameras()
        sys_info.append(f"Picamera2 detected: YES, cameras: {cam_ids}")
        for cam_id in cam_ids:
            ok_open, err_open = picam.open(cam_id)
            if not ok_open:
                results.append(TestResult(
                    backend="picamera2",
                    device=f"cam_id={cam_id}",
                    resolution=(0, 0),
                    fps_req=0,
                    opened=False,
                    configured=False,
                    fps_measured=0.0,
                    frame_shape="",
                    frame_dtype="",
                    sample_path="",
                    error=err_open
                ))
                continue

            for size in resolutions:
                for fps in fps_list:
                    ok_cfg, err_cfg = picam.configure(size, fps)
                    if not ok_cfg:
                        results.append(TestResult(
                            backend="picamera2",
                            device=f"cam_id={cam_id}",
                            resolution=size,
                            fps_req=fps,
                            opened=True,
                            configured=False,
                            fps_measured=0.0,
                            frame_shape="",
                            frame_dtype="",
                            sample_path="",
                            error=err_cfg
                        ))
                        continue

                    fps_meas, last, err = measure_fps_grab(picam.get_frame, args.frames)
                    shape = str(getattr(last, "shape", "None"))
                    dtype = str(getattr(last, "dtype", ""))
                    sample_path = ""
                    if last is not None:
                        sample_file = out_dir / "samples" / f"picamera2_cam{cam_id}_{size[0]}x{size[1]}_{fps}fps.jpg"
                        if safe_imwrite(sample_file, last):
                            sample_path = str(sample_file.relative_to(out_dir))

                        if args.show and have_display():
                            cv2.imshow("picamera2 sample", last)
                            cv2.waitKey(50)

                    results.append(TestResult(
                        backend="picamera2",
                        device=f"cam_id={cam_id}",
                        resolution=size,
                        fps_req=fps,
                        opened=True,
                        configured=True,
                        fps_measured=fps_meas,
                        frame_shape=shape,
                        frame_dtype=dtype,
                        sample_path=sample_path,
                        error=err
                    ))

            picam.close()
    else:
        sys_info.append("Picamera2 detected: NO (module not importable)")
    sys_info.append("")

    # -------- OpenCV V4L2 tests --------
    v4l2_devs = v4l2.list_devices(args.max_v4l2)
    sys_info.append(f"OpenCV V4L2 devices: {v4l2_devs if v4l2_devs else 'none'}")

    for dev in v4l2_devs:
        for size in resolutions:
            for fps in fps_list:
                opened, configured, fps_meas, last, err = v4l2.run_test(dev, size, fps, args.frames)
                shape = str(getattr(last, "shape", "None"))
                dtype = str(getattr(last, "dtype", ""))
                sample_path = ""
                if last is not None:
                    last_bgr = to_bgr(last)
                    sample_file = out_dir / "samples" / f"v4l2_dev{dev}_{size[0]}x{size[1]}_{fps}fps.jpg"
                    if safe_imwrite(sample_file, last_bgr):
                        sample_path = str(sample_file.relative_to(out_dir))
                    if args.show and have_display():
                        cv2.imshow("v4l2 sample", last_bgr)
                        cv2.waitKey(50)

                results.append(TestResult(
                    backend="opencv_v4l2",
                    device=f"index={dev}",
                    resolution=size,
                    fps_req=fps,
                    opened=opened,
                    configured=configured,
                    fps_measured=fps_meas,
                    frame_shape=shape,
                    frame_dtype=dtype,
                    sample_path=sample_path,
                    error=err
                ))
    sys_info.append("")

    # -------- OpenCV GStreamer tests --------
    sys_info.append(f"OpenCV GStreamer backend: {'enabled' if gst.enabled else 'disabled'}")
    if gst.enabled:
        # libcamerasrc обычно один (CSI), поэтому без device id
        for size in resolutions:
            for fps in fps_list:
                opened, configured, fps_meas, last, err = gst.run_test(size, fps, args.frames)
                shape = str(getattr(last, "shape", "None"))
                dtype = str(getattr(last, "dtype", ""))
                sample_path = ""
                if last is not None:
                    last_bgr = to_bgr(last)
                    sample_file = out_dir / "samples" / f"gst_libcamerasrc_{size[0]}x{size[1]}_{fps}fps.jpg"
                    if safe_imwrite(sample_file, last_bgr):
                        sample_path = str(sample_file.relative_to(out_dir))
                    if args.show and have_display():
                        cv2.imshow("gst sample", last_bgr)
                        cv2.waitKey(50)

                results.append(TestResult(
                    backend="opencv_gstreamer_libcamerasrc",
                    device="libcamerasrc",
                    resolution=size,
                    fps_req=fps,
                    opened=opened,
                    configured=configured,
                    fps_measured=fps_meas,
                    frame_shape=shape,
                    frame_dtype=dtype,
                    sample_path=sample_path,
                    error=err
                ))
    sys_info.append("")

    # Close preview windows
    if args.show and have_display():
        cv2.destroyAllWindows()

    # ------------------------------ write report ------------------------------
    def fmt_bool(b): return "YES" if b else "NO"

    lines = []
    lines.extend(sys_info)
    lines.append("==== RESULTS ====")
    lines.append("Legend: opened/configured = did backend open/configure; fps_measured = measured by grabbing frames")
    lines.append("")

    # Group results by backend/device
    results_sorted = sorted(results, key=lambda r: (r.backend, r.device, r.resolution[0]*10000 + r.resolution[1], r.fps_req))

    current_group = None
    for r in results_sorted:
        grp = (r.backend, r.device)
        if grp != current_group:
            current_group = grp
            lines.append(f"\n--- {r.backend} | {r.device} ---")

        res_str = f"{r.resolution[0]}x{r.resolution[1]}" if r.resolution != (0, 0) else "-"
        lines.append(
            f"  {res_str:>10} @ {r.fps_req:>3} fps | "
            f"opened={fmt_bool(r.opened)} configured={fmt_bool(r.configured)} | "
            f"measured={r.fps_measured:6.2f} fps | "
            f"frame={r.frame_shape} {r.frame_dtype} | "
            f"sample={r.sample_path or '-'} | "
            f"err={r.error or '-'}"
        )

    # Find best modes (top by measured fps) per backend/device
    lines.append("\n==== BEST MODES (by measured FPS) ====")
    from collections import defaultdict
    best = defaultdict(lambda: None)

    for r in results:
        if not r.opened or not r.configured or r.error:
            continue
        key = (r.backend, r.device)
        cur = best[key]
        if cur is None or r.fps_measured > cur.fps_measured:
            best[key] = r

    if not best:
        lines.append("No successful modes found.")
    else:
        for (backend, device), r in sorted(best.items(), key=lambda x: (x[0][0], x[0][1])):
            lines.append(
                f"- {backend} | {device}: {r.resolution[0]}x{r.resolution[1]} @ {r.fps_req} fps "
                f"-> measured {r.fps_measured:.2f} fps, sample {r.sample_path or '-'}"
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDONE. Report saved to: {report_path}")
    print(f"Samples saved under: {out_dir / 'samples'}")


if __name__ == "__main__":
    main()
