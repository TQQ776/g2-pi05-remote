#!/usr/bin/env python3
"""读 G2 的三路相机：存图 + 测实际出帧率。

用法（必须先 source GDK 的 env.sh）：
    source /mnt/models/gdk_x86/app/env.sh
    python3.10 snap.py                      # 存一组图到 ./snaps/<时间戳>/
    python3.10 snap.py --out /tmp/cam --rounds 20   # 顺带测帧率
    python3.10 snap.py --no-save --rounds 30        # 只测帧率不落盘

帧率注意：**不要紧轮询**。高频调用会饿死 GDK 的接收线程，实测 2 ms 轮询只能
拿到 2.4 Hz，正常用法（每次等新帧）是 28.8 Hz。所以这里每轮都等新时间戳。
"""

from __future__ import annotations

import argparse
import time

CAMERAS = (
    ("base_0_rgb", "kHeadColor"),
    ("left_wrist_0_rgb", "kHandLeftColor"),
    ("right_wrist_0_rgb", "kHandRightColor"),
)


def main() -> int:
    p = argparse.ArgumentParser(description="G2 相机快照 / 帧率测量")
    p.add_argument("--out", default="snaps", help="存图目录")
    p.add_argument("--rounds", type=int, default=1, help="取多少组图；>1 时同时统计帧率")
    p.add_argument("--timeout", type=float, default=2.0, help="等新帧的超时（秒）")
    p.add_argument("--no-save", action="store_true", help="只测速，不写文件")
    args = p.parse_args()

    import agibot_gdk as gdk  # noqa: PLC0415  只在装了 GDK 的机器上可用

    gdk.gdk_init()
    cam = gdk.Camera()          # 默认订阅全部相机；只订阅三路可减少负载
    time.sleep(2.0)             # 订阅要时间，别在 Camera() 之前 sleep
    kinds = [(key, getattr(gdk.CameraType, name)) for key, name in CAMERAS]

    outdir = None
    if not args.no_save:
        from pathlib import Path  # noqa: PLC0415

        outdir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
        outdir.mkdir(parents=True, exist_ok=True)

    last = {}
    arrivals = {key: [] for key, _ in kinds}
    for r in range(args.rounds):
        deadline = time.monotonic() + args.timeout
        for key, kind in kinds:
            while True:
                image = cam.get_latest_image(kind, 50.0)
                ts = image.timestamp_ns
                if last.get(key) != ts or time.monotonic() >= deadline:
                    break
                time.sleep(0.02)   # 不要低于一帧周期，否则会饿死接收线程
            last[key] = ts
            arrivals[key].append(time.monotonic())
            h, w = image.height, image.width
            enc = image.encoding
            if outdir is not None:
                suffix = "jpg" if "JPEG" in str(enc) else "png" if "PNG" in str(enc) else "raw"
                (outdir / f"{r:03d}_{key}.{suffix}").write_bytes(bytes(image.data))
            print(f"[{r:03d}] {key:16s} {w}x{h}  enc={enc}  "
                  f"{len(bytes(image.data)) / 1024:7.1f} KB", flush=True)

    print("\n各相机 GDK 自报帧率:", {k: round(cam.get_image_fps(kind), 1) for k, kind in kinds})

    if args.rounds > 1:
        for key, _ in kinds:
            a = arrivals[key]
            if len(a) < 3:
                continue
            total = a[-1] - a[0]
            print(f"  {key:16s} {len(a) - 1} 组 / {total:.2f}s = {(len(a) - 1) / total:5.1f} Hz")

    if outdir is not None:
        print(f"\n已保存到 {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
