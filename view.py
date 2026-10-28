#!/usr/bin/env python3
"""三路相机本地实时显示。

和 `snap.py` 的区别：这个是一直刷新的窗口，用来肉眼确认「画面是不是真的在动」。
取图走的是 `G2IO.read_images`，也就是 client.py 实际用的那条路径 ——
所以窗口里看到的帧龄、新帧率，就是控制时模型真正拿到的数据质量。

只读：不发任何运动指令。

用法（必须先有 GDK 环境，用 run_view.sh 最省事）：
    ./run_view.sh              # 三路一起看
    ./run_view.sh --scale 2    # 窗口放大一倍
"""

from __future__ import annotations

import argparse
import io as _io
import signal
import threading
import time
import tkinter as tk
from hashlib import blake2b
from tkinter import ttk

from PIL import Image, ImageTk

from g2_io import CAMERA_KEYS, G2IO

TITLES = {
    "base_0_rgb": "头部彩色",
    "left_wrist_0_rgb": "左腕",
    "right_wrist_0_rgb": "右腕",
}

STALE_S = 0.10  # 帧龄超过这个值就标红：控制时这就是「模型看到的是旧画面」
RATE_WINDOW = 1.0  # 新帧率统计窗口（秒）


class Reader(threading.Thread):
    """后台持续取图。tkinter 只能在主线程碰，所以这里只写共享字典。"""

    def __init__(self, io_: G2IO):
        super().__init__(daemon=True)
        self.io = io_
        self.stop = threading.Event()
        self.lock = threading.Lock()

        self.images: dict = {}  # 相机键 -> PIL 图（已缩略）
        self.version: dict = {k: 0 for k in CAMERA_KEYS}  # 主线程靠它判断要不要重画
        self.info: dict = {k: "" for k in CAMERA_KEYS}
        self.stale: dict = {k: False for k in CAMERA_KEYS}
        self.rates: dict = {k: 0.0 for k in CAMERA_KEYS}

        self.error: str | None = None
        self.capture_ms = 0.0
        self.loop_hz = 0.0

        self._hash: dict = {}
        self._new = {k: 0 for k in CAMERA_KEYS}
        self._rate_t0 = time.monotonic()
        self._loops = 0
        self._loop_t0 = time.monotonic()

    def run(self) -> None:
        while not self.stop.is_set():
            t0 = time.monotonic()
            try:
                blobs, shapes = self.io.read_images(92, 0.05)
            except Exception as exc:  # noqa: BLE001  取图失败不该让窗口静默卡死
                with self.lock:
                    self.error = str(exc)
                time.sleep(0.2)
                continue

            capture_ms = (time.monotonic() - t0) * 1000.0
            for key in CAMERA_KEYS:
                blob = blobs[key]
                digest = blake2b(blob, digest_size=8).digest()
                with self.lock:
                    if self._hash.get(key) != digest:
                        self._hash[key] = digest
                        self._new[key] += 1

                image = Image.open(_io.BytesIO(blob))
                image.draft("RGB", (480, 360))  # JPEG 解码时直接降采样，快一个数量级
                image = image.convert("RGB")
                image.thumbnail((420, 320), Image.Resampling.BILINEAR)

                age = self.io.last_frame_age.get(key)
                stale = age is None or age > STALE_S
                h, w = shapes[key]
                info = f"{w}×{h}  {len(blob) / 1024:.0f} KB  " + (
                    "帧龄 —" if age is None else f"帧龄 {age * 1000:.0f} ms"
                )

                with self.lock:
                    self.images[key] = image
                    self.version[key] += 1
                    self.info[key] = info
                    self.stale[key] = stale
                    self.error = None

            with self.lock:
                self.capture_ms = capture_ms
                self._loops += 1
                now = time.monotonic()
                loop_span = now - self._loop_t0
                if loop_span >= RATE_WINDOW:
                    self.loop_hz = self._loops / loop_span
                    span = now - self._rate_t0
                    self.rates = {k: self._new[k] / span for k in CAMERA_KEYS}
                    self._new = {k: 0 for k in CAMERA_KEYS}
                    self._rate_t0 = now
                    self._loop_t0 = now
                    self._loops = 0

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "images": dict(self.images),
                "version": dict(self.version),
                "info": dict(self.info),
                "stale": dict(self.stale),
                "rates": dict(self.rates),
                "error": self.error,
                "capture_ms": self.capture_ms,
                "loop_hz": self.loop_hz,
            }


class Viewer:
    def __init__(self, root: tk.Tk, reader: Reader):
        self.root = root
        self.reader = reader
        self.shown: dict = {}

        root.title("智元 G2 相机监视器（G2IO 取图路径）")
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.canvas_labels: dict = {}
        self.info_labels: dict = {}
        for col, key in enumerate(CAMERA_KEYS):
            box = ttk.LabelFrame(root, text=TITLES[key])
            box.grid(row=0, column=col, padx=6, pady=6, sticky="nsew")
            canvas = ttk.Label(box, text="等待图像…", anchor="center", width=42)
            canvas.pack(padx=4, pady=4)
            info = ttk.Label(box, text="", anchor="center")
            info.pack(pady=(0, 6))
            self.canvas_labels[key] = canvas
            self.info_labels[key] = info
            root.columnconfigure(col, weight=1)
        root.rowconfigure(0, weight=1)

        self.status = ttk.Label(root, text="正在连接 G2…", anchor="w", relief="sunken")
        self.status.grid(row=1, column=0, columnspan=len(CAMERA_KEYS), sticky="ew", padx=6, pady=6)

        root.after(66, self.refresh)

    def refresh(self) -> None:
        snap = self.reader.snapshot()

        for key in CAMERA_KEYS:
            version = snap["version"].get(key, 0)
            if key in snap["images"] and self.shown.get(key) != version:
                photo = ImageTk.PhotoImage(snap["images"][key])
                label = self.canvas_labels[key]
                label.configure(image=photo, text="")
                label.image = photo  # 必须留引用，否则会被 GC 掉变白框
                self.shown[key] = version

            # 帧龄超限时标红 —— 带图片的 Label 设 foreground 看不出来，只能标在信息行上
            stale = snap["stale"].get(key)
            info = snap["info"].get(key, "")
            self.info_labels[key].configure(
                text=f"⚠ 陈旧  {info}" if stale else info,
                foreground="red" if stale else "",
            )

        if snap["error"]:
            self.status.configure(text=f"取图失败：{snap['error']}", foreground="red")
        else:
            rates = "  ".join(f"{TITLES[k][:2]} {snap['rates'].get(k, 0):.1f}" for k in CAMERA_KEYS)
            self.status.configure(
                text=f"采图 {snap['capture_ms']:.1f} ms · 循环 {snap['loop_hz']:.1f} Hz · "
                f"新帧率 {rates} Hz",
                foreground="",
            )

        self.root.after(66, self.refresh)

    def close(self) -> None:
        self.reader.stop.set()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=1.0, help="tk 界面缩放（1.0 为默认）")
    args = parser.parse_args()

    root = tk.Tk()
    if args.scale != 1.0:
        try:
            root.tk.call("tk", "scaling", args.scale * float(root.tk.call("tk", "scaling")))
        except tk.TclError:
            pass

    # G2IO 初始化要等 2 秒（G2IO.__init__ 里的 init_timeout），
    # 先弹个窗口，否则这 2 秒里点了运行看不到任何反应，会以为失败了
    splash = ttk.Label(root, text="正在连接 G2（约 2 秒）…", padding=40)
    splash.pack()
    root.update()

    io_ = G2IO()
    splash.destroy()

    reader = Reader(io_)
    reader.start()
    Viewer(root, reader)

    # Tk 的 mainloop 会把 SIGINT 变成某个回调里的 KeyboardInterrupt 然后继续跑，
    # 所以 Ctrl-C 默认关不掉窗口，必须自己接信号再 quit
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: root.quit())

    try:
        root.mainloop()
    finally:
        reader.stop.set()
        io_.close()


if __name__ == "__main__":
    main()
