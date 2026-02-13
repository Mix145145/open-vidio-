#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi 4B + Sony IMX477 — ArUco-сканер (DICT_4X4_50; решётка 3 × 3)

▪ калибровка: движение только по X (±20 мм, шаг 2 мм) —
  останавливается при ≥ 4 найденных метках; результат
  сохраняется в aruco_calib.json;

▪ «смарт-шаг» 0.80 × FOV  → ≈ 20 % перекрытия;

▪ склейка только MultiBand (простой стабильный вариант).
"""
import importlib
import json
import logging
import math
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import serial
import serial.tools.list_ports
from PySide6 import QtCore, QtGui, QtWidgets


# ─────────── константы ────────────────────────────────────────────────
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_MM = 5.3  # сторона чёрного квадрата, мм
CAL_Z_MM = 83
STEP_FACTOR = 0.80  # смарт-шаг = 0.80 × FOV  (≈ 20 %)
CONFIG_FILE = "config.json"
DEFAULT_RESOLUTION = "2028x1520"
RESOLUTION_PRESETS = {
    "FHD": "1920x1080",
    "2K": "2560x1440",
    "4K": "3840x2160",
}
CENTER_X = 54
CENTER_Y = 110


try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass
logging.getLogger("cv2").setLevel(logging.ERROR)


# ─────────── утилиты ──────────────────────────────────────────────────
def f(value, default=0.0):
    try:
        if isinstance(value, QtCore.QObject) and hasattr(value, "text"):
            return float(str(value.text()).replace(",", "."))
        if isinstance(value, QtCore.QObject) and hasattr(value, "currentText"):
            return float(str(value.currentText()).replace(",", "."))
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def parse_resolution(value, fallback=(1920, 1080)):
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except Exception:
        return fallback


def get_picamera2_class():
    if importlib.util.find_spec("picamera2") is None:
        return None
    return importlib.import_module("picamera2").Picamera2


class CameraManager:
    def __init__(self, logger):
        self.logger = logger
        self.picamera2_cls = get_picamera2_class()
        self.picam = None
        self.picam_id = None
        self.picam_resolution = None
        self.lock = threading.Lock()
        self.libcamera_available = self._detect_libcamera_tools()
        self.libcamera_error_reported = False

    def _detect_libcamera_tools(self):
        try:
            return subprocess.run(
                ["libcamera-still", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=2,
            ).returncode == 0
        except Exception:
            return False

    def list_cameras(self):
        if self.picamera2_cls:
            return self._list_cameras_picamera2()
        if self.libcamera_available:
            return self._list_cameras_libcamera()
        return self._list_video_devices_opencv()

    def _list_cameras_libcamera(self):
        try:
            proc = subprocess.run(
                ["libcamera-hello", "--list-cameras"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            cameras = []
            for line in text.splitlines():
                if "/" in line and ":" in line and "-" in line:
                    cam_id = str(len(cameras))
                    cameras.append((cam_id, f"Camera {cam_id}: {line.strip()}"))
            return cameras or [("0", "Camera 0: libcamera")]
        except Exception:
            return [("0", "Camera 0: libcamera")]

    def _list_cameras_picamera2(self):
        infos = self.picamera2_cls.global_camera_info()
        out = []
        for idx, info in enumerate(infos):
            model = info.get("Model", f"Camera {idx}")
            out.append((str(idx), f"Camera {idx}: {model}"))
        if not out:
            out = [("0", "Camera 0")]
        return out

    def _list_video_devices_opencv(self):
        devices = []
        for idx in range(10):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                devices.append((str(idx), f"/dev/video{idx}"))
            cap.release()
        return devices or [("0", "/dev/video0")]

    def _ensure_picam_config(self, requested_size):
        candidates = [
            requested_size,
            (2028, 1520),
            (1920, 1080),
            (1332, 990),
            (4056, 3040),
        ]
        last_err = None
        for size in candidates:
            try:
                config = self.picam.create_still_configuration(main={"size": size})
                self.picam.configure(config)
                self.picam.start()
                time.sleep(0.15)
                return size
            except Exception as exc:
                last_err = exc
                try:
                    self.picam.stop()
                except Exception:
                    pass
        raise RuntimeError(f"Cannot configure Picamera2. Last error: {last_err}")

    def _ensure_picam(self, cam_id, resolution):
        if self.picam and self.picam_id != cam_id:
            self.picam.close()
            self.picam = None
        if not self.picam:
            cam_index = int(cam_id)
            self.picam = self.picamera2_cls(cam_index)
            self.picam_id = cam_id
            self.picam_resolution = None
        if self.picam_resolution != resolution:
            try:
                self.picam.stop()
            except Exception:
                pass
            self.picam_resolution = self._ensure_picam_config(resolution)

    def snap(self, cam_id, width=1920, height=1080):
        with self.lock:
            if self.picamera2_cls:
                try:
                    self._ensure_picam(cam_id, (width, height))
                    frame = self.picam.capture_array()
                    if frame is None:
                        return None
                    if frame.ndim == 3 and frame.shape[2] >= 3:
                        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    return frame
                except Exception as exc:
                    self.logger.info("Picamera2 snap failed: %s", exc)
                    if not self.libcamera_available:
                        return None
            if self.libcamera_available:
                frame = self._snap_libcamera(cam_id, width, height)
                if frame is not None:
                    return frame
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
            cam_arg = int(cam_id) if str(cam_id).isdigit() else cam_id
            if isinstance(cam_arg, int) and os.name != "nt" and not os.path.exists(f"/dev/video{cam_arg}"):
                return None
            cap = cv2.VideoCapture(cam_arg, backend)
            cap.set(3, width)
            cap.set(4, height)
            for _ in range(3):
                cap.grab()
            ok, frame = cap.read()
            cap.release()
            return frame if ok else None

    def warm_up(self, cam_id, width=1920, height=1080):
        with self.lock:
            if self.picamera2_cls:
                try:
                    self._ensure_picam(cam_id, (width, height))
                    return True
                except Exception as exc:
                    self.logger.info("Picamera2 warm up failed: %s", exc)
                    if not self.libcamera_available:
                        return False
            if self.libcamera_available:
                return self._warm_up_libcamera(cam_id, width, height)
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
            cam_arg = int(cam_id) if str(cam_id).isdigit() else cam_id
            if isinstance(cam_arg, int) and os.name != "nt" and not os.path.exists(f"/dev/video{cam_arg}"):
                return False
            cap = cv2.VideoCapture(cam_arg, backend)
            if not cap.isOpened():
                return False
            cap.set(3, width)
            cap.set(4, height)
            cap.read()
            cap.release()
            return True

    def set_exposure(self, cam_id, exposure_us, resolution=None):
        exposure = max(1, int(exposure_us))
        with self.lock:
            if self.picamera2_cls:
                if resolution is None:
                    resolution = self.picam_resolution or parse_resolution(DEFAULT_RESOLUTION)
                try:
                    self._ensure_picam(cam_id, resolution)
                    self.picam.set_controls({"ExposureTime": exposure})
                    return True
                except Exception as exc:
                    self.logger.info("Picamera2 exposure failed: %s", exc)
                    if not self.libcamera_available:
                        return False
            if self.libcamera_available:
                # Для libcamera-still экспозиция задаётся на уровне команды захвата.
                return True
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
            cam_arg = int(cam_id) if str(cam_id).isdigit() else cam_id
            if isinstance(cam_arg, int) and os.name != "nt" and not os.path.exists(f"/dev/video{cam_arg}"):
                return False
            cap = cv2.VideoCapture(cam_arg, backend)
            if not cap.isOpened():
                return False
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            cap.release()
            return True

    def _warm_up_libcamera(self, cam_id, width, height):
        frame = self._snap_libcamera(cam_id, width, height, timeout_ms=600)
        return frame is not None

    def _snap_libcamera(self, cam_id, width, height, timeout_ms=250):
        cmd = [
            "libcamera-still",
            "-n",
            "--camera",
            str(cam_id),
            "--width",
            str(width),
            "--height",
            str(height),
            "--timeout",
            str(timeout_ms),
            "--encoding",
            "jpg",
            "-o",
            "-",
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=max(3, timeout_ms / 1000 + 2),
            )
            if proc.returncode != 0 or not proc.stdout:
                if not self.libcamera_error_reported:
                    self.logger.info("libcamera-still failed: %s", (proc.stderr or b"").decode(errors="ignore").strip())
                    self.libcamera_error_reported = True
                return None
            arr = np.frombuffer(proc.stdout, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as exc:
            if not self.libcamera_error_reported:
                pretty_cmd = " ".join(shlex.quote(c) for c in cmd)
                self.logger.info("libcamera capture failed (%s): %s", pretty_cmd, exc)
                self.libcamera_error_reported = True
            return None

    def close(self):
        with self.lock:
            if self.picam:
                try:
                    self.picam.stop()
                except Exception:
                    pass
                self.picam.close()
                self.picam = None
                self.picam_id = None
                self.picam_resolution = None




class LocalControlServer:
    def __init__(self, scanner, logger, host="0.0.0.0", port=8765):
        self.scanner = scanner
        self.logger = logger
        self.host = host
        self.port = port
        self.httpd = None
        self.thread = None

    def start(self):
        scanner = self.scanner
        logger = self.logger

        class Handler(BaseHTTPRequestHandler):
            def _json(self, code, payload):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/status":
                    self._json(200, {
                        "serial_connected": bool(scanner.ser and scanner.ser.is_open),
                        "serial_port": scanner.comb_ports.currentText() if hasattr(scanner, "comb_ports") else "",
                        "camera": scanner.cam_combo.currentData() if hasattr(scanner, "cam_combo") else "0",
                        "resolution": scanner.res_combo.currentText() if hasattr(scanner, "res_combo") else DEFAULT_RESOLUTION,
                    })
                    return
                if parsed.path == "/gcode":
                    cmd = parse_qs(parsed.query).get("cmd", [""])[0].strip()
                    if not cmd:
                        self._json(400, {"ok": False, "error": "missing cmd"})
                        return
                    ok = scanner._g(cmd)
                    self._json(200, {"ok": bool(ok), "cmd": cmd})
                    return
                self._json(404, {"ok": False, "error": "not found"})

            def log_message(self, format, *args):
                logger.info("LAN API: " + format, *args)

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.logger.info("LAN API started on http://%s:%s", self.host, self.port)

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

class QtLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)


class GridView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_cols = 0
        self.grid_rows = 0
        self.fov_x = 1.0
        self.fov_y = 1.0
        self.thumbs = {}

    def set_grid(self, cols, rows, fov_x, fov_y):
        self.grid_cols = cols
        self.grid_rows = rows
        self.fov_x = max(fov_x, 0.01)
        self.fov_y = max(fov_y, 0.01)
        self.update()

    def clear_thumbs(self):
        self.thumbs.clear()
        self.update()

    def set_thumb(self, col, row, image):
        self.thumbs[(col, row)] = image
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#0e1b2b"))
        if not self.grid_cols or not self.grid_rows:
            return
        width = self.width()
        height = self.height()
        aspect = self.fov_x / self.fov_y
        cell_w = min(width / self.grid_cols, height * aspect / self.grid_rows)
        cell_h = cell_w / aspect
        total_w = cell_w * self.grid_cols
        total_h = cell_h * self.grid_rows
        offset_x = (width - total_w) / 2
        offset_y = (height - total_h) / 2

        pen = QtGui.QPen(QtGui.QColor("#444"))
        painter.setPen(pen)
        for i in range(self.grid_cols + 1):
            x = offset_x + i * cell_w
            painter.drawLine(QtCore.QPointF(x, offset_y), QtCore.QPointF(x, offset_y + total_h))
        for j in range(self.grid_rows + 1):
            y = offset_y + j * cell_h
            painter.drawLine(QtCore.QPointF(offset_x, y), QtCore.QPointF(offset_x + total_w, y))

        for (col, row), image in self.thumbs.items():
            inv_row = self.grid_rows - 1 - row
            x0 = offset_x + col * cell_w
            y0 = offset_y + inv_row * cell_h
            pixmap = QtGui.QPixmap.fromImage(image)
            if pixmap.isNull():
                continue
            pixmap = pixmap.scaled(
                int(cell_w), int(cell_h),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            px = x0 + (cell_w - pixmap.width()) / 2
            py = y0 + (cell_h - pixmap.height()) / 2
            painter.drawPixmap(QtCore.QPointF(px, py), pixmap)


class FocusPreviewDialog(QtWidgets.QDialog):
    def __init__(self, parent, camera_manager, cam_id, resolution, exposure_us=10000):
        super().__init__(parent)
        self.setWindowTitle("Предпросмотр камеры")
        self.camera_manager = camera_manager
        self.cam_id = cam_id
        self.resolution = resolution
        self.exposure_us = max(1, int(exposure_us))
        self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 360)
        self.warned_no_frame = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(QtWidgets.QLabel("Экспозиция (мкс)"))
        self.exposure_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exposure_slider.setRange(100, 200000)
        self.exposure_slider.setValue(self.exposure_us)
        self.exposure_spin = QtWidgets.QSpinBox()
        self.exposure_spin.setRange(100, 200000)
        self.exposure_spin.setValue(self.exposure_us)
        self.exposure_slider.valueChanged.connect(self.exposure_spin.setValue)
        self.exposure_spin.valueChanged.connect(self.exposure_slider.setValue)
        self.exposure_spin.valueChanged.connect(self._apply_exposure)
        controls_layout.addWidget(self.exposure_slider)
        controls_layout.addWidget(self.exposure_spin)
        layout.addLayout(controls_layout)
        self._apply_exposure(self.exposure_us)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(200)

    def _apply_exposure(self, value):
        self.exposure_us = value
        self.camera_manager.set_exposure(self.cam_id, value, self.resolution)

    def _update_frame(self):
        frame = self.camera_manager.snap(self.cam_id, *self.resolution)
        if frame is None:
            self.image_label.setText("Нет кадра")
            self.camera_manager.logger.info("Preview: empty frame")
            if not self.warned_no_frame:
                QtWidgets.QMessageBox.warning(self, "Preview", "Камера не вернула кадр")
                self.warned_no_frame = True
            return
        image = cv_to_qimage(frame)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


def cv_to_qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


# ─────────── главное окно ─────────────────────────────────────────────
class Scanner(QtWidgets.QMainWindow):
    log_signal = QtCore.Signal(str)
    progress_signal = QtCore.Signal(float)
    thumb_signal = QtCore.Signal(QtGui.QImage, int, int)
    notify_signal = QtCore.Signal(str, str, str)
    scan_finished_signal = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tuposcan ArUco scanner")
        self.resize(1280, 960)

        self.logger = logging.getLogger("tuposcan")
        self.logger.setLevel(logging.INFO)
        self.log_signal.connect(self._append_log)
        handler = QtLogHandler(self.log_signal)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        self.logger.addHandler(handler)

        self.camera_manager = CameraManager(self.logger)
        self.local_server = LocalControlServer(self, self.logger)

        self.com = ""
        self.cam = "0"
        self.resolution = DEFAULT_RESOLUTION
        self.fovX = "30"
        self.fovY = "17"
        self.stepX = "30"
        self.stepY = "17"
        self.z = str(CAL_Z_MM)
        self.feed = "1500"
        self.scan_profile = ""
        self.focus_profile = ""
        self.scan_name = ""
        self.scan_width = ""
        self.scan_height = ""
        self.focus_name = ""
        self.focus_z = ""
        self.focus_fovX = ""
        self.focus_fovY = ""
        self.focus_step = "1"
        self.exposure_us = "10000"
        self.fov_move_equals_step = True
        self.fov_profiles = {
            RESOLUTION_PRESETS["FHD"]: {"fovX": 30.0, "fovY": 17.0},
            RESOLUTION_PRESETS["2K"]: {"fovX": 30.0, "fovY": 17.0},
            RESOLUTION_PRESETS["4K"]: {"fovX": 30.0, "fovY": 17.0},
        }
        self.default_overlap_mm = 4.0
        self.slice_width_px = 12
        self.slice_offset_px = 0
        self.capture_fps = 30.0
        self.y_start = 0.0
        self.x_start = 0.0
        self.stabilization_enabled = True

        self.ser = None
        self.frames = []
        self.grid_cols = 0
        self.grid_rows = 0
        self.positions = []
        self.stitched = None
        self.scan_stop_requested = False
        self.last_scan_dir = None

        self._load_config()
        self._build_ui()

        self.progress_signal.connect(self._set_progress)
        self.thumb_signal.connect(self._add_thumb)
        self.notify_signal.connect(self._show_message)
        self.scan_finished_signal.connect(self._scan_finished)
        QtCore.QTimer.singleShot(0, self._auto_connect_camera)
        QtCore.QTimer.singleShot(0, self._auto_connect_serial)
        self.local_server.start()

    # ─────────── сохранение/загрузка калибровки ───────────
    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                self.fovX = f"{data.get('fovX', float(self.fovX)):.2f}"
                self.fovY = f"{data.get('fovY', float(self.fovY)):.2f}"
                if data.get("resolution"):
                    self.resolution = data["resolution"]
                self.scan_profiles = data.get("scan_profiles", {})
                self.focus_profiles = data.get("focus_profiles", {})
                self.scan_profile = data.get("selected_scan_profile", "")
                self.focus_profile = data.get("selected_focus_profile", "")
                loaded_fov_profiles = data.get("fov_profiles", {})
                for res, values in loaded_fov_profiles.items():
                    self.fov_profiles[res] = {
                        "fovX": float(values.get("fovX", self.fov_profiles.get(res, {}).get("fovX", 30.0))),
                        "fovY": float(values.get("fovY", self.fov_profiles.get(res, {}).get("fovY", 17.0))),
                    }
                self.fov_move_equals_step = bool(data.get("fov_move_equals_step", True))
                self.default_overlap_mm = float(data.get("default_overlap_mm", self.default_overlap_mm))
                self.slice_width_px = int(data.get("slice_width_px", self.slice_width_px))
                self.slice_offset_px = int(data.get("slice_offset_px", self.slice_offset_px))
                self.capture_fps = float(data.get("capture_fps", self.capture_fps))
                self.y_start = float(data.get("y_start", self.y_start))
                self.x_start = float(data.get("x_start", self.x_start))
                self.stabilization_enabled = bool(data.get("stabilization_enabled", self.stabilization_enabled))
                self._apply_fov_profile_for_resolution(self.resolution)
            except Exception:
                self.logger.info("Не удалось прочитать конфиг, использую значения по умолчанию")
        if not hasattr(self, "scan_profiles"):
            self.scan_profiles = {
                "Весь стол": {"width": 200, "height": 200},
                "Плата 95x95": {"width": 95, "height": 95},
            }
        if not hasattr(self, "focus_profiles"):
            self.focus_profiles = {
                "Стандарт": {"z": 85, "fovX": 30, "fovY": 17},
            }
        if not self.scan_profile:
            self.scan_profile = next(iter(self.scan_profiles))
        if not self.focus_profile:
            self.focus_profile = next(iter(self.focus_profiles))
        self._load_profile_values()

    def _save_config(self):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "fovX": float(self.fovX),
                        "fovY": float(self.fovY),
                        "resolution": self.resolution,
                        "scan_profiles": self.scan_profiles,
                        "focus_profiles": self.focus_profiles,
                        "selected_scan_profile": self.scan_profile,
                        "selected_focus_profile": self.focus_profile,
                        "fov_profiles": self.fov_profiles,
                        "fov_move_equals_step": self.fov_move_equals_step,
                        "default_overlap_mm": self.default_overlap_mm,
                        "slice_width_px": self.slice_width_px,
                        "slice_offset_px": self.slice_offset_px,
                        "capture_fps": self.capture_fps,
                        "x_start": self.x_start,
                        "y_start": self.y_start,
                        "stabilization_enabled": self.stabilization_enabled,
                    },
                    fp,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            self.logger.info("Не удалось сохранить конфиг")

    # ─────────── UI ───────────
    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        top_bar = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_bar)
        top_bar.addWidget(QtWidgets.QLabel("Сериал порт"))
        self.comb_ports = QtWidgets.QComboBox()
        self._refresh_ports()
        top_bar.addWidget(self.comb_ports)
        btn_refresh = QtWidgets.QPushButton("Обновить")
        btn_refresh.clicked.connect(self._refresh_ports)
        top_bar.addWidget(btn_refresh)
        btn_connect = QtWidgets.QPushButton("Подключить")
        btn_connect.clicked.connect(self._connect)
        top_bar.addWidget(btn_connect)
        btn_home = QtWidgets.QPushButton("Home")
        btn_home.clicked.connect(lambda: self._g("G28"))
        top_bar.addWidget(btn_home)
        btn_unlock = QtWidgets.QPushButton("Unlock")
        btn_unlock.clicked.connect(lambda: self._g("M17"))
        top_bar.addWidget(btn_unlock)
        top_bar.addStretch(1)

        tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(tabs)

        scan_tab = QtWidgets.QWidget()
        tabs.addTab(scan_tab, "Сканирование")
        scan_layout = QtWidgets.QVBoxLayout(scan_tab)

        settings_row = QtWidgets.QHBoxLayout()
        scan_layout.addLayout(settings_row)

        settings_row.addWidget(QtWidgets.QLabel("Камера"))
        self.cam_combo = QtWidgets.QComboBox()
        self._populate_cameras()
        settings_row.addWidget(self.cam_combo)
        settings_row.addWidget(QtWidgets.QLabel("Разрешение"))
        self.res_combo = QtWidgets.QComboBox()
        self.res_combo.addItems([RESOLUTION_PRESETS["FHD"], RESOLUTION_PRESETS["2K"], RESOLUTION_PRESETS["4K"], "2028x1520", "1332x990", "4056x3040"])
        if self.resolution not in [self.res_combo.itemText(i) for i in range(self.res_combo.count())]:
            self.res_combo.addItem(self.resolution)
        self.res_combo.setCurrentText(self.resolution)
        self.res_combo.currentTextChanged.connect(self._on_resolution_changed)
        settings_row.addWidget(self.res_combo)
        settings_row.addWidget(QtWidgets.QLabel("FOV X,Y"))
        self.fovx_edit = QtWidgets.QLineEdit(self.fovX)
        self.fovy_edit = QtWidgets.QLineEdit(self.fovY)
        self.fovx_edit.setFixedWidth(60)
        self.fovy_edit.setFixedWidth(60)
        settings_row.addWidget(self.fovx_edit)
        settings_row.addWidget(self.fovy_edit)
        settings_row.addWidget(QtWidgets.QLabel("Step X,Y"))
        self.stepx_edit = QtWidgets.QLineEdit(self.stepX)
        self.stepy_edit = QtWidgets.QLineEdit(self.stepY)
        self.stepx_edit.setFixedWidth(60)
        self.stepy_edit.setFixedWidth(60)
        settings_row.addWidget(self.stepx_edit)
        settings_row.addWidget(self.stepy_edit)
        settings_row.addWidget(QtWidgets.QLabel("Z"))
        self.z_edit = QtWidgets.QLineEdit(self.z)
        self.z_edit.setFixedWidth(60)
        settings_row.addWidget(self.z_edit)
        settings_row.addWidget(QtWidgets.QLabel("F"))
        self.feed_edit = QtWidgets.QLineEdit(self.feed)
        self.feed_edit.setFixedWidth(60)
        settings_row.addWidget(self.feed_edit)
        settings_row.addStretch(1)

        strip_row = QtWidgets.QHBoxLayout()
        scan_layout.addLayout(strip_row)
        strip_row.addWidget(QtWidgets.QLabel("X0,Y0"))
        self.scan_x0_edit = QtWidgets.QLineEdit(f"{self.x_start:.2f}")
        self.scan_y0_edit = QtWidgets.QLineEdit(f"{self.y_start:.2f}")
        self.scan_x0_edit.setFixedWidth(70)
        self.scan_y0_edit.setFixedWidth(70)
        strip_row.addWidget(self.scan_x0_edit)
        strip_row.addWidget(self.scan_y0_edit)
        strip_row.addWidget(QtWidgets.QLabel("Overlap X (мм)"))
        self.overlap_edit = QtWidgets.QLineEdit(f"{self.default_overlap_mm:.2f}")
        self.overlap_edit.setFixedWidth(70)
        strip_row.addWidget(self.overlap_edit)
        strip_row.addWidget(QtWidgets.QLabel("slice px"))
        self.slice_width_edit = QtWidgets.QLineEdit(str(self.slice_width_px))
        self.slice_width_edit.setFixedWidth(60)
        strip_row.addWidget(self.slice_width_edit)
        strip_row.addWidget(QtWidgets.QLabel("slice offset"))
        self.slice_offset_edit = QtWidgets.QLineEdit(str(self.slice_offset_px))
        self.slice_offset_edit.setFixedWidth(60)
        strip_row.addWidget(self.slice_offset_edit)
        strip_row.addWidget(QtWidgets.QLabel("FPS"))
        self.fps_edit = QtWidgets.QLineEdit(f"{self.capture_fps:.1f}")
        self.fps_edit.setFixedWidth(60)
        strip_row.addWidget(self.fps_edit)
        self.stab_checkbox = QtWidgets.QCheckBox("Стабилизация X")
        self.stab_checkbox.setChecked(self.stabilization_enabled)
        strip_row.addWidget(self.stab_checkbox)
        strip_row.addStretch(1)

        control_row = QtWidgets.QHBoxLayout()
        scan_layout.addLayout(control_row)
        self.btn_scan = QtWidgets.QPushButton("Scan")
        self.btn_scan.clicked.connect(self._start_scan)
        control_row.addWidget(self.btn_scan)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop_scan)
        control_row.addWidget(self.btn_stop)
        btn_save = QtWidgets.QPushButton("Save")
        btn_save.clicked.connect(self._save)
        control_row.addWidget(btn_save)
        control_row.addStretch(1)

        self.progress = QtWidgets.QProgressBar()
        scan_layout.addWidget(self.progress)

        self.grid_view = GridView()
        scan_layout.addWidget(self.grid_view, stretch=1)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        scan_layout.addWidget(self.log_view)

        settings_tab = QtWidgets.QWidget()
        tabs.addTab(settings_tab, "Настройки")
        settings_layout = QtWidgets.QVBoxLayout(settings_tab)

        scan_group = QtWidgets.QGroupBox("Профили сканирования")
        settings_layout.addWidget(scan_group)
        scan_group_layout = QtWidgets.QHBoxLayout(scan_group)
        scan_group_layout.addWidget(QtWidgets.QLabel("Профиль"))
        self.scan_profiles_box = QtWidgets.QComboBox()
        self.scan_profiles_box.addItems(list(self.scan_profiles))
        self.scan_profiles_box.setCurrentText(self.scan_profile)
        self.scan_profiles_box.currentTextChanged.connect(self._select_scan_profile)
        scan_group_layout.addWidget(self.scan_profiles_box)
        scan_group_layout.addWidget(QtWidgets.QLabel("Имя"))
        self.scan_name_edit = QtWidgets.QLineEdit(self.scan_name)
        scan_group_layout.addWidget(self.scan_name_edit)
        scan_group_layout.addWidget(QtWidgets.QLabel("Ширина"))
        self.scan_width_edit = QtWidgets.QLineEdit(self.scan_width)
        self.scan_width_edit.setFixedWidth(70)
        scan_group_layout.addWidget(self.scan_width_edit)
        scan_group_layout.addWidget(QtWidgets.QLabel("Высота"))
        self.scan_height_edit = QtWidgets.QLineEdit(self.scan_height)
        self.scan_height_edit.setFixedWidth(70)
        scan_group_layout.addWidget(self.scan_height_edit)
        btn_scan_save = QtWidgets.QPushButton("Сохранить")
        btn_scan_save.clicked.connect(self._save_scan_profile)
        scan_group_layout.addWidget(btn_scan_save)
        btn_scan_delete = QtWidgets.QPushButton("Удалить")
        btn_scan_delete.clicked.connect(self._delete_scan_profile)
        scan_group_layout.addWidget(btn_scan_delete)

        focus_group = QtWidgets.QGroupBox("Профили фокуса")
        settings_layout.addWidget(focus_group)
        focus_group_layout = QtWidgets.QHBoxLayout(focus_group)
        focus_group_layout.addWidget(QtWidgets.QLabel("Профиль"))
        self.focus_profiles_box = QtWidgets.QComboBox()
        self.focus_profiles_box.addItems(list(self.focus_profiles))
        self.focus_profiles_box.setCurrentText(self.focus_profile)
        self.focus_profiles_box.currentTextChanged.connect(self._select_focus_profile)
        focus_group_layout.addWidget(self.focus_profiles_box)
        focus_group_layout.addWidget(QtWidgets.QLabel("Имя"))
        self.focus_name_edit = QtWidgets.QLineEdit(self.focus_name)
        focus_group_layout.addWidget(self.focus_name_edit)
        focus_group_layout.addWidget(QtWidgets.QLabel("Z"))
        self.focus_z_edit = QtWidgets.QLineEdit(self.focus_z)
        self.focus_z_edit.setFixedWidth(60)
        focus_group_layout.addWidget(self.focus_z_edit)
        focus_group_layout.addWidget(QtWidgets.QLabel("FOV"))
        self.focus_fovx_edit = QtWidgets.QLineEdit(self.focus_fovX)
        self.focus_fovy_edit = QtWidgets.QLineEdit(self.focus_fovY)
        self.focus_fovx_edit.setFixedWidth(60)
        self.focus_fovy_edit.setFixedWidth(60)
        focus_group_layout.addWidget(self.focus_fovx_edit)
        focus_group_layout.addWidget(self.focus_fovy_edit)
        focus_group_layout.addWidget(QtWidgets.QLabel("Шаг Z"))
        self.focus_step_edit = QtWidgets.QLineEdit(self.focus_step)
        self.focus_step_edit.setFixedWidth(60)
        focus_group_layout.addWidget(self.focus_step_edit)
        btn_z_up = QtWidgets.QPushButton("Z +")
        btn_z_up.clicked.connect(lambda: self._nudge_z(1))
        focus_group_layout.addWidget(btn_z_up)
        btn_z_down = QtWidgets.QPushButton("Z -")
        btn_z_down.clicked.connect(lambda: self._nudge_z(-1))
        focus_group_layout.addWidget(btn_z_down)
        btn_focus_save = QtWidgets.QPushButton("Сохранить")
        btn_focus_save.clicked.connect(self._save_focus_profile)
        focus_group_layout.addWidget(btn_focus_save)
        btn_focus_delete = QtWidgets.QPushButton("Удалить")
        btn_focus_delete.clicked.connect(self._delete_focus_profile)
        focus_group_layout.addWidget(btn_focus_delete)
        btn_focus_check = QtWidgets.QPushButton("Проверить фокус")
        btn_focus_check.clicked.connect(self._check_focus)
        focus_group_layout.addWidget(btn_focus_check)
        btn_focus_preview = QtWidgets.QPushButton("Открыть предпросмотр")
        btn_focus_preview.clicked.connect(self._open_focus_preview)
        focus_group_layout.addWidget(btn_focus_preview)
        focus_group_layout.addWidget(QtWidgets.QLabel("Экспозиция (мкс)"))
        self.exposure_edit = QtWidgets.QLineEdit(self.exposure_us)
        self.exposure_edit.setFixedWidth(80)
        self.exposure_edit.editingFinished.connect(self._apply_exposure_setting)
        focus_group_layout.addWidget(self.exposure_edit)
        focus_group_layout.addStretch(1)

        fov_group = QtWidgets.QGroupBox("Настройки FOV")
        settings_layout.addWidget(fov_group)
        fov_layout = QtWidgets.QHBoxLayout(fov_group)
        fov_layout.addWidget(QtWidgets.QLabel("Режим"))
        self.fov_res_mode_combo = QtWidgets.QComboBox()
        self.fov_res_mode_combo.addItems(list(RESOLUTION_PRESETS.keys()))
        fov_layout.addWidget(self.fov_res_mode_combo)
        fov_layout.addWidget(QtWidgets.QLabel("FOV X,Y (мм)"))
        self.fov_profile_x_edit = QtWidgets.QLineEdit(self.fovX)
        self.fov_profile_y_edit = QtWidgets.QLineEdit(self.fovY)
        self.fov_profile_x_edit.setFixedWidth(70)
        self.fov_profile_y_edit.setFixedWidth(70)
        fov_layout.addWidget(self.fov_profile_x_edit)
        fov_layout.addWidget(self.fov_profile_y_edit)
        self.fov_move_checkbox = QtWidgets.QCheckBox("FOV = Move (шаг = FOV)")
        self.fov_move_checkbox.setChecked(self.fov_move_equals_step)
        fov_layout.addWidget(self.fov_move_checkbox)
        btn_fov_apply = QtWidgets.QPushButton("Сохранить FOV")
        btn_fov_apply.clicked.connect(self._save_active_fov_profile)
        fov_layout.addWidget(btn_fov_apply)
        btn_fov_center = QtWidgets.QPushButton("В центр на высоту")
        btn_fov_center.clicked.connect(self._move_to_fov_center)
        fov_layout.addWidget(btn_fov_center)
        fov_layout.addStretch(1)

        self.fov_res_mode_combo.currentTextChanged.connect(self._on_fov_mode_changed)
        self.fov_move_checkbox.toggled.connect(self._on_fov_move_toggled)

        settings_layout.addStretch(1)

    # ─────────── Serial ───────────
    def _connect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            port = self.comb_ports.currentText()
            self.ser = serial.Serial(port, 250000, timeout=1)
            time.sleep(2)
            self.ser.reset_input_buffer()
            self._info("Serial", "Connected")
            self.logger.info("Serial connected: %s", port)
        except Exception as exc:
            self._error("Serial", str(exc))

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.comb_ports.clear()
        self.comb_ports.addItems(ports)
        if not ports:
            return
        preferred = "/dev/ttyUSB0"
        index = self.comb_ports.findText(preferred)
        self.comb_ports.setCurrentIndex(index if index >= 0 else 0)

    def _g(self, cmd):
        if not (self.ser and self.ser.is_open):
            return False
        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\n").encode())
        self.ser.flush()
        while True:
            line = self.ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith("ok"):
                return True
            if "error" in line.lower():
                self.logger.info("G-code error: %s", line)
                return False

    # ─────────── smart-step ───────────
    def _apply_steps(self):
        if self.fov_move_equals_step:
            self.stepX = f"{float(self.fovX):.2f}"
            self.stepY = f"{float(self.fovY):.2f}"
            return
        k = STEP_FACTOR
        self.stepX = f"{float(self.fovX) * k:.2f}"
        self.stepY = f"{float(self.fovY) * k:.2f}"

    def _resolution_key_from_mode(self, mode_name):
        return RESOLUTION_PRESETS.get(mode_name, self.res_combo.currentText())

    def _mode_from_resolution(self, resolution):
        for mode, res in RESOLUTION_PRESETS.items():
            if res == resolution:
                return mode
        return "FHD"

    def _apply_fov_profile_for_resolution(self, resolution):
        profile = self.fov_profiles.get(resolution)
        if not profile:
            profile = {"fovX": float(self.fovX), "fovY": float(self.fovY)}
            self.fov_profiles[resolution] = profile
        self.fovX = f"{float(profile.get('fovX', self.fovX)):.2f}"
        self.fovY = f"{float(profile.get('fovY', self.fovY)):.2f}"
        self._apply_steps()

    def _save_active_fov_profile(self):
        mode = self.fov_res_mode_combo.currentText()
        resolution = self._resolution_key_from_mode(mode)
        fx = f(self.fov_profile_x_edit.text(), f(self.fovx_edit.text(), 30.0))
        fy = f(self.fov_profile_y_edit.text(), f(self.fovy_edit.text(), 17.0))
        if fx <= 0 or fy <= 0:
            self._warn("FOV", "FOV должен быть > 0")
            return
        self.fov_profiles[resolution] = {"fovX": fx, "fovY": fy}
        self.res_combo.setCurrentText(resolution)
        self._apply_fov_profile_for_resolution(resolution)
        self._sync_fields()
        self._save_config()
        self.logger.info("FOV profile saved for %s: %.2f x %.2f", resolution, fx, fy)

    def _on_fov_mode_changed(self, mode):
        resolution = self._resolution_key_from_mode(mode)
        self.res_combo.setCurrentText(resolution)

    def _on_resolution_changed(self, resolution):
        self.resolution = resolution
        self._apply_fov_profile_for_resolution(resolution)
        if hasattr(self, "fov_res_mode_combo"):
            mode = self._mode_from_resolution(resolution)
            self.fov_res_mode_combo.blockSignals(True)
            self.fov_res_mode_combo.setCurrentText(mode)
            self.fov_res_mode_combo.blockSignals(False)
        self._sync_fields()

    def _on_fov_move_toggled(self, checked):
        self.fov_move_equals_step = bool(checked)
        self._apply_steps()
        self._sync_fields()
        self._save_config()

    def _move_to_fov_center(self):
        if not (self.ser and self.ser.is_open):
            self._warn("Serial", "not connected")
            return
        z = f(self.focus_z_edit.text(), CAL_Z_MM)
        self.z_edit.setText(f"{z:.2f}")
        for cmd in (
            "G90",
            "M400",
            f"G1 X{CENTER_X:.2f} Y{CENTER_Y:.2f} Z{z:.2f} F{int(f(self.feed_edit.text(), 1500))}",
            "M400",
        ):
            if not self._g(cmd):
                return
        self.logger.info("FOV setup position: center at Z=%s", z)

    # ─────────── калибровка ───────────
    def _start_calibration(self):
        threading.Thread(target=self._calibrate, daemon=True).start()

    def _calibrate(self):
        try:
            feed = int(f(self.feed_edit.text(), 1500))
            z = f(self.z_edit.text(), CAL_Z_MM)
            cam = self.cam_combo.currentData() or "0"
            for cmd in ("G90", "G28", "M400", f"G1 Z{z:.2f} F{feed}", "M400"):
                if not self._g(cmd):
                    return

            dict4 = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
            params = cv2.aruco.DetectorParameters()
            ids_seen = set()
            px = []
            cv2.namedWindow("Calib", cv2.WINDOW_NORMAL)

            for dx in range(-20, 22, 2):
                self._g(f"G1 X{dx:.2f} Y0 F{feed}")
                self._g("M400")
                time.sleep(0.25)
                res = parse_resolution(self.res_combo.currentText(), (1920, 1080))
                frame = self.camera_manager.snap(cam, *res)
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, dict4, parameters=params)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
                    for corner, id_val in zip(corners, ids.flatten()):
                        ids_seen.add(int(id_val))
                        pts = corner.reshape(4, 2)
                        side = np.mean([
                            np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)
                        ])
                        px.append(side)
                        xc, yc = pts.mean(0)
                        cv2.putText(
                            frame,
                            str(id_val),
                            (int(xc) - 7, int(yc) - 7),
                            0,
                            0.5,
                            (0, 255, 0),
                            1,
                        )
                cv2.putText(
                    frame,
                    f"seen {len(ids_seen)}",
                    (10, frame.shape[0] - 15),
                    0,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow("Calib", frame)
                cv2.waitKey(1)
                if len(ids_seen) >= 4:
                    break
            cv2.destroyWindow("Calib")

            if len(px) < 4:
                raise RuntimeError("мало меток")

            pxmm = np.median(px) / MARKER_MM
            res = parse_resolution(self.res_combo.currentText(), (1920, 1080))
            frame = self.camera_manager.snap(cam, *res)
            if frame is None:
                raise RuntimeError("камера не вернула кадр")
            h_px, w_px = frame.shape[:2]
            fx, fy = w_px / pxmm, h_px / pxmm
            self.fovX = f"{fx:.2f}"
            self.fovY = f"{fy:.2f}"
            self._apply_steps()
            self._sync_fields()
            self._save_config()
            self._info("Calibration", f"FOV  {fx:.2f} × {fy:.2f} мм\nmarkers: {len(ids_seen)}")
            self.logger.info("Calibration done: %s x %s", fx, fy)
        except Exception as exc:
            self._error("Calibration", str(exc))
            self.logger.info("Calibration error: %s", exc)

    # ─────────── профили ───────────
    def _load_profile_values(self):
        scan = self.scan_profiles.get(self.scan_profile)
        if scan:
            self.scan_name = self.scan_profile
            self.scan_width = str(scan.get("width", ""))
            self.scan_height = str(scan.get("height", ""))
        focus = self.focus_profiles.get(self.focus_profile)
        if focus:
            self.focus_name = self.focus_profile
            self.focus_z = str(focus.get("z", ""))
            self.focus_fovX = str(focus.get("fovX", ""))
            self.focus_fovY = str(focus.get("fovY", ""))
            self.z = f"{float(focus.get('z', self.z)):.2f}"
            self._apply_fov_profile_for_resolution(self.resolution)

    def _select_scan_profile(self, name):
        self.scan_profile = name
        self._load_profile_values()
        self._sync_fields()
        self._save_config()

    def _select_focus_profile(self, name):
        self.focus_profile = name
        self._load_profile_values()
        self._sync_fields()
        self._save_config()

    def _save_scan_profile(self):
        name = self.scan_name_edit.text().strip()
        if not name:
            self._warn("Профиль", "Введите имя профиля")
            return
        width = f(self.scan_width_edit.text())
        height = f(self.scan_height_edit.text())
        if width <= 0 or height <= 0:
            self._warn("Профиль", "Размеры должны быть > 0")
            return
        self.scan_profiles[name] = {"width": width, "height": height}
        self.scan_profile = name
        self.scan_profiles_box.clear()
        self.scan_profiles_box.addItems(list(self.scan_profiles))
        self.scan_profiles_box.setCurrentText(name)
        self._save_config()
        self.logger.info("Scan profile saved: %s", name)

    def _delete_scan_profile(self):
        name = self.scan_profile
        if name in self.scan_profiles:
            del self.scan_profiles[name]
            self.scan_profile = next(iter(self.scan_profiles), "")
            self.scan_profiles_box.clear()
            self.scan_profiles_box.addItems(list(self.scan_profiles))
            self.scan_profiles_box.setCurrentText(self.scan_profile)
            self._load_profile_values()
            self._sync_fields()
            self._save_config()
            self.logger.info("Scan profile deleted: %s", name)

    def _save_focus_profile(self):
        name = self.focus_name_edit.text().strip()
        if not name:
            self._warn("Профиль", "Введите имя профиля")
            return
        z = f(self.focus_z_edit.text())
        self.focus_profiles[name] = {"z": z}
        self.focus_profile = name
        self.focus_profiles_box.clear()
        self.focus_profiles_box.addItems(list(self.focus_profiles))
        self.focus_profiles_box.setCurrentText(name)
        self._load_profile_values()
        self._sync_fields()
        self._save_config()
        self.logger.info("Focus profile saved: %s", name)

    def _delete_focus_profile(self):
        name = self.focus_profile
        if name in self.focus_profiles:
            del self.focus_profiles[name]
            self.focus_profile = next(iter(self.focus_profiles), "")
            self.focus_profiles_box.clear()
            self.focus_profiles_box.addItems(list(self.focus_profiles))
            self.focus_profiles_box.setCurrentText(self.focus_profile)
            self._load_profile_values()
            self._sync_fields()
            self._save_config()
            self.logger.info("Focus profile deleted: %s", name)

    def _nudge_z(self, sign):
        if not (self.ser and self.ser.is_open):
            self._warn("Serial", "not connected")
            return
        step = f(self.focus_step_edit.text(), 1.0)
        z = f(self.focus_z_edit.text(), 0.0) + (step * sign)
        self.focus_z_edit.setText(f"{z:.2f}")
        self.z_edit.setText(f"{z:.2f}")
        self._g("G90")
        self._g(f"G1 X{CENTER_X:.2f} Y{CENTER_Y:.2f} Z{z:.2f} F{int(f(self.feed_edit.text(), 1500))}")
        self.logger.info("Nudge Z to %s", z)

    def _check_focus(self):
        if not (self.ser and self.ser.is_open):
            self._warn("Serial", "not connected")
            return
        z = f(self.focus_z_edit.text(), CAL_Z_MM)
        self.z_edit.setText(f"{z:.2f}")
        for cmd in (
            "G90",
            "G28",
            "M400",
            f"G1 X{CENTER_X:.2f} Y{CENTER_Y:.2f} Z{z:.2f} F{int(f(self.feed_edit.text(), 1500))}",
            "M400",
        ):
            if not self._g(cmd):
                return
        self.logger.info("Focus check at Z=%s", z)

    def _open_focus_preview(self):
        cam = self.cam_combo.currentData() or "0"
        res = parse_resolution(self.res_combo.currentText(), (1920, 1080))
        dialog = FocusPreviewDialog(
            self,
            self.camera_manager,
            cam,
            res,
            f(self.exposure_edit.text(), 10000),
        )
        dialog.exec()

    def _apply_exposure_setting(self):
        cam = self.cam_combo.currentData() or "0"
        res = parse_resolution(self.res_combo.currentText(), (1920, 1080))
        exposure = f(self.exposure_edit.text(), 10000)
        self.exposure_us = f"{exposure:.0f}"
        self.camera_manager.set_exposure(cam, exposure, res)

    # ─────────── сетка ───────────
    def _build_scan_plan(self):
        prof = self.scan_profiles.get(self.scan_profile, {"width": 0, "height": 0})
        x_size = max(0.0, float(prof.get("width", 0.0)))
        y_size = max(0.0, float(prof.get("height", 0.0)))
        x0 = f(self.scan_x0_edit.text(), 0.0)
        y0 = f(self.scan_y0_edit.text(), 0.0)
        fov_x = f(self.fovx_edit.text(), 30.0)
        fov_y = f(self.fovy_edit.text(), 17.0)
        overlap = f(self.overlap_edit.text(), self.default_overlap_mm)
        overlap = max(0.1, min(fov_x * 0.9, overlap))
        step_x = max(0.1, fov_x - overlap)
        strips = 1 if x_size <= fov_x else int(math.ceil((x_size - fov_x) / step_x) + 1)
        x_positions = [x0 + i * step_x for i in range(strips)]
        y_end = y0 + y_size
        return {
            "x0": x0,
            "y0": y0,
            "x_size": x_size,
            "y_size": y_size,
            "y_end": y_end,
            "fov_x": fov_x,
            "fov_y": fov_y,
            "overlap": overlap,
            "step_x": step_x,
            "strips": strips,
            "x_positions": x_positions,
        }

    def _build_grid(self):
        plan = self._build_scan_plan()
        self.grid_cols = plan["strips"]
        self.grid_rows = 1
        self.positions = [(i, 0, x, plan["y0"]) for i, x in enumerate(plan["x_positions"])]
        self.grid_view.set_grid(self.grid_cols, 1, plan["fov_x"], max(plan["y_size"], plan["fov_y"]))
        self.grid_view.clear_thumbs()

    def _stop_scan(self):
        self.scan_stop_requested = True
        self.logger.info("Stop requested")

    def _record_strip_video(self, cam, resolution, out_path, feed, x, y0, y1, fps):
        cap = cv2.VideoCapture(int(cam) if str(cam).isdigit() else cam, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        if real_fps <= 1:
            real_fps = fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, real_fps, resolution)
        if not writer.isOpened() or not cap.isOpened():
            cap.release()
            writer.release()
            raise RuntimeError("Не удалось открыть видео для записи")

        if not self._g(f"G1 X{x:.2f} Y{y0:.2f} F{feed}"):
            raise RuntimeError("Ошибка позиционирования перед полосой")
        if not self._g("M400"):
            raise RuntimeError("M400 error")

        distance = abs(y1 - y0)
        mm_per_sec = max(0.1, feed / 60.0)
        duration = distance / mm_per_sec

        if not self._g(f"G1 X{x:.2f} Y{y1:.2f} F{feed}"):
            raise RuntimeError("Ошибка движения по Y")

        frames_written = 0
        started = time.time()
        while (time.time() - started) <= (duration + 0.5):
            ok, frame = cap.read()
            if ok:
                writer.write(frame)
                frames_written += 1
            if self.scan_stop_requested:
                break
        self._g("M400")
        cap.release()
        writer.release()
        return max(1, frames_written), float(real_fps)

    def _extract_strip_from_video(self, video_path, strip_path, y_distance_mm, slice_width_px, slice_offset_px, stabilize):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"Видео {video_path.name} не содержит кадров")

        h, w = frames[0].shape[:2]
        slice_width_px = max(1, min(w // 3, int(slice_width_px)))
        center = w // 2 + int(slice_offset_px)
        left = max(0, min(w - slice_width_px, center - slice_width_px // 2))
        right = left + slice_width_px

        strips = []
        ref_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        ref_patch = ref_gray[:, left:right]
        for frame in frames:
            x_shift = 0
            if stabilize:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                patch = gray[:, left:right]
                r = cv2.matchTemplate(patch, ref_patch, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(r)
                x_shift = max_loc[0]
            l = max(0, min(w - slice_width_px, left + x_shift))
            sl = frame[:, l:l + slice_width_px].mean(axis=1)
            strips.append(sl.astype(np.uint8))

        strip_img = np.stack(strips, axis=0)
        target_h = max(1, int(round(y_distance_mm * len(strips) / max(0.1, y_distance_mm))))
        if target_h != strip_img.shape[0]:
            strip_img = cv2.resize(strip_img, (strip_img.shape[1], target_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(strip_path), strip_img)
        return strip_img

    def _align_and_blend_strips(self, strip_images, overlap_px):
        base = strip_images[0].copy()
        for nxt in strip_images[1:]:
            ov = min(overlap_px, base.shape[1] - 2, nxt.shape[1] - 2)
            ov = max(2, ov)
            left = base[:, -ov:]
            right = nxt[:, :ov]
            best_shift = 0
            best_score = -1e9
            for shift in range(-40, 41):
                if shift >= 0:
                    a = left[shift:]
                    b = right[:right.shape[0] - shift]
                else:
                    a = left[:left.shape[0] + shift]
                    b = right[-shift:]
                if a.shape[0] < 30:
                    continue
                score = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)) * -1.0
                if score > best_score:
                    best_score = score
                    best_shift = shift
            if best_shift > 0:
                nxt_adj = np.pad(nxt, ((best_shift, 0), (0, 0), (0, 0)), mode="edge")
                base_adj = np.pad(base, ((0, best_shift), (0, 0), (0, 0)), mode="edge")
            elif best_shift < 0:
                nxt_adj = np.pad(nxt, ((0, -best_shift), (0, 0), (0, 0)), mode="edge")
                base_adj = np.pad(base, ((-best_shift, 0), (0, 0), (0, 0)), mode="edge")
            else:
                nxt_adj = nxt
                base_adj = base
            alpha = np.linspace(0, 1, ov, dtype=np.float32)[None, :, None]
            blend_zone = (base_adj[:, -ov:].astype(np.float32) * (1 - alpha) + nxt_adj[:, :ov].astype(np.float32) * alpha).astype(np.uint8)
            base = np.concatenate([base_adj[:, :-ov], blend_zone, nxt_adj[:, ov:]], axis=1)
        return base

    # ─────────── сканирование ───────────
    def _start_scan(self):
        if self.btn_scan:
            self.btn_scan.setEnabled(False)
        self.scan_stop_requested = False
        threading.Thread(target=self._scan, daemon=True).start()

    def _scan(self):
        try:
            if not (self.ser and self.ser.is_open):
                self._notify("warn", "Serial", "not connected")
                return
            self._build_grid()
            plan = self._build_scan_plan()
            feed = int(f(self.feed_edit.text(), 1500))
            z = f(self.z_edit.text(), CAL_Z_MM)
            cam = self.cam_combo.currentData() or "0"
            res = parse_resolution(self.res_combo.currentText(), (1920, 1080))
            fps = max(1.0, f(self.fps_edit.text(), self.capture_fps))
            slice_width = max(1, int(f(self.slice_width_edit.text(), self.slice_width_px)))
            slice_offset = int(f(self.slice_offset_edit.text(), self.slice_offset_px))
            stabilize = self.stab_checkbox.isChecked()

            self.default_overlap_mm = plan["overlap"]
            self.capture_fps = fps
            self.slice_width_px = slice_width
            self.slice_offset_px = slice_offset
            self.x_start = plan["x0"]
            self.y_start = plan["y0"]
            self.stabilization_enabled = stabilize
            self._save_config()

            for cmd in ("G90", "M400", f"G1 Z{z:.2f} F{feed}", "M400"):
                if not self._g(cmd):
                    return

            scan_dir, _ = self._prepare_scan_output()
            self.last_scan_dir = scan_dir
            raw_dir = scan_dir / "raw"
            strips_dir = scan_dir / "strips"
            result_dir = scan_dir / "result"
            raw_dir.mkdir(exist_ok=True)
            strips_dir.mkdir(exist_ok=True)
            result_dir.mkdir(exist_ok=True)

            strip_images = []
            report = {
                "resolution": self.res_combo.currentText(),
                "scan": plan,
                "fps_requested": fps,
                "slice_width_px": slice_width,
                "slice_offset_px": slice_offset,
                "stabilization": stabilize,
                "strips": [],
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            total = max(1, plan["strips"])
            for i, x in enumerate(plan["x_positions"]):
                if self.scan_stop_requested:
                    break
                self.logger.info("Полоса %s/%s: X=%.2f", i + 1, total, x)
                video_path = raw_dir / f"strip_{i:03d}.mp4"
                strip_path = strips_dir / f"strip_{i:03d}.png"
                frames, real_fps = self._record_strip_video(cam, res, video_path, feed, x, plan["y0"], plan["y_end"], fps)
                strip_img = self._extract_strip_from_video(
                    video_path,
                    strip_path,
                    plan["y_size"],
                    slice_width,
                    slice_offset,
                    stabilize,
                )
                thumb = cv2.resize(strip_img, (320, 180), interpolation=cv2.INTER_AREA)
                self.thumb_signal.emit(cv_to_qimage(thumb), i, 0)
                strip_images.append(strip_img)
                report["strips"].append({
                    "index": i,
                    "x_mm": x,
                    "video": str(video_path.relative_to(scan_dir)),
                    "strip": str(strip_path.relative_to(scan_dir)),
                    "frames": frames,
                    "fps_actual": real_fps,
                })
                self.progress_signal.emit((i + 1) / total)

            if not strip_images:
                raise RuntimeError("Нет данных полос для сборки")

            ppx = strip_images[0].shape[1] / max(0.1, plan["fov_x"])
            overlap_px = int(round(plan["overlap"] * ppx))
            final_img = self._align_and_blend_strips(strip_images, overlap_px)
            final_path = result_dir / "final.png"
            cv2.imwrite(str(final_path), final_img)
            self.stitched = final_img
            report["result"] = {
                "file": str(final_path.relative_to(scan_dir)),
                "shape": list(final_img.shape),
                "overlap_px": overlap_px,
            }
            report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(scan_dir / "report.json", "w", encoding="utf-8") as fp:
                json.dump(report, fp, indent=2, ensure_ascii=False)
            self.logger.info("Скан завершен: %s", final_path)
            self._notify("info", "Scan", f"done: {final_path}")
        except Exception as exc:
            self._notify("error", "Scan", str(exc))
            self.logger.info("Scan error: %s", exc)
        finally:
            self.scan_finished_signal.emit()

    # ─────────── thumb/grid ───────────
    def _add_thumb(self, image, col, row):
        self.grid_view.set_thumb(col, row, image)

    # ─────────── save ───────────
    def _save(self):
        if self.stitched is None:
            self._notify("warn", "Result", "Нет итогового изображения")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if path:
            cv2.imwrite(path, self.stitched)
            self._notify("info", "Saved", path)
            self.logger.info("Saved panorama to %s", path)

    def _sync_fields(self):
        self.fovx_edit.setText(self.fovX)
        self.fovy_edit.setText(self.fovY)
        self.stepx_edit.setText(self.stepX)
        self.stepy_edit.setText(self.stepY)
        self.z_edit.setText(self.z)
        self.scan_name_edit.setText(self.scan_name)
        self.scan_width_edit.setText(self.scan_width)
        self.scan_height_edit.setText(self.scan_height)
        self.focus_name_edit.setText(self.focus_name)
        self.focus_z_edit.setText(self.focus_z)
        self.focus_fovx_edit.setText(self.focus_fovX)
        self.focus_fovy_edit.setText(self.focus_fovY)
        self.focus_step_edit.setText(self.focus_step)
        self.exposure_edit.setText(self.exposure_us)
        if hasattr(self, "scan_x0_edit"):
            self.scan_x0_edit.setText(f"{self.x_start:.2f}")
            self.scan_y0_edit.setText(f"{self.y_start:.2f}")
            self.overlap_edit.setText(f"{self.default_overlap_mm:.2f}")
            self.slice_width_edit.setText(str(self.slice_width_px))
            self.slice_offset_edit.setText(str(self.slice_offset_px))
            self.fps_edit.setText(f"{self.capture_fps:.1f}")
            self.stab_checkbox.setChecked(self.stabilization_enabled)
        if hasattr(self, "fov_profile_x_edit"):
            mode = self._mode_from_resolution(self.res_combo.currentText())
            self.fov_res_mode_combo.blockSignals(True)
            self.fov_res_mode_combo.setCurrentText(mode)
            self.fov_res_mode_combo.blockSignals(False)
            active_res = self._resolution_key_from_mode(mode)
            profile = self.fov_profiles.get(active_res, {"fovX": f(self.fovX), "fovY": f(self.fovY)})
            self.fov_profile_x_edit.setText(f"{float(profile.get('fovX', self.fovX)):.2f}")
            self.fov_profile_y_edit.setText(f"{float(profile.get('fovY', self.fovY)):.2f}")
            self.fov_move_checkbox.setChecked(self.fov_move_equals_step)
        self.stepx_edit.setReadOnly(self.fov_move_equals_step)
        self.stepy_edit.setReadOnly(self.fov_move_equals_step)

    def _populate_cameras(self):
        self.cam_combo.clear()
        for cam_id, label in self.camera_manager.list_cameras():
            self.cam_combo.addItem(label, cam_id)
        if self.cam_combo.count():
            preferred = -1
            for i in range(self.cam_combo.count()):
                if "imx477" in self.cam_combo.itemText(i).lower():
                    preferred = i
                    break
            if preferred < 0:
                preferred = self.cam_combo.findData("0")
            self.cam_combo.setCurrentIndex(preferred if preferred >= 0 else 0)

    def _auto_connect_serial(self):
        port = self.comb_ports.currentText().strip()
        if not port:
            return
        self._connect()

    def _auto_connect_camera(self):
        cam = self.cam_combo.currentData() or "0"
        res = parse_resolution(self.res_combo.currentText(), (1920, 1080))
        ok = self.camera_manager.warm_up(cam, *res)
        if ok:
            self.logger.info("Camera ready: %s @ %sx%s", cam, res[0], res[1])
        else:
            self.logger.info("Camera warm up failed")

    def _set_progress(self, value):
        self.progress.setValue(int(value * 100))

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def _notify(self, level, title, message):
        self.notify_signal.emit(level, title, message)

    def _info(self, title, message):
        self._notify("info", title, message)

    def _warn(self, title, message):
        self._notify("warn", title, message)

    def _error(self, title, message):
        self._notify("error", title, message)

    def _show_message(self, level, title, message):
        if level == "warn":
            QtWidgets.QMessageBox.warning(self, title, message)
        elif level == "error":
            QtWidgets.QMessageBox.critical(self, title, message)
        else:
            QtWidgets.QMessageBox.information(self, title, message)

    def _scan_finished(self):
        if self.btn_scan:
            self.btn_scan.setEnabled(True)
        self.scan_stop_requested = False

    def _prepare_scan_output(self):
        base = Path("scans")
        base.mkdir(exist_ok=True)
        name = self.scan_name_edit.text().strip() or "scan"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        scan_dir = base / f"{name}_{timestamp}"
        scan_dir.mkdir(parents=True, exist_ok=True)
        return scan_dir, None

    def closeEvent(self, event):
        self._save_config()
        self.camera_manager.close()
        self.local_server.stop()
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().closeEvent(event)


# ─────────── run ───────────
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QWidget {
            background-color: #0e1b2b;
            color: #e0e6f0;
        }
        QGroupBox {
            border: 1px solid #2a3d55;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {
            background-color: #13243a;
            border: 1px solid #2a3d55;
            padding: 4px;
        }
        QPushButton {
            background-color: #1c3550;
            border: 1px solid #2a3d55;
            padding: 6px 10px;
        }
        QPushButton:hover {
            background-color: #244465;
        }
        QProgressBar {
            background-color: #13243a;
            border: 1px solid #2a3d55;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #2e5d86;
        }
        QTabBar::tab {
            background-color: #13243a;
            padding: 6px 12px;
            border: 1px solid #2a3d55;
        }
        QTabBar::tab:selected {
            background-color: #1c3550;
        }
        """
    )
    window = Scanner()
    window.show()
    app.exec()
