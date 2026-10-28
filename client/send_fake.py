#!/usr/bin/env python3
"""不接机器人也能测通服务器：用合成图像和零状态发请求。

在开发机上跑（只需要 numpy + PIL），用来验证服务器地址、协议、模型加载是否正常。

    python send_fake.py --host 1.2.3.4 --port 8765 --rounds 5
"""

from __future__ import annotations

import argparse
import io
import socket
import statistics
import time

import numpy as np

import wire

# 和训练数据一致的相机分辨率
SHAPES = {"base_0_rgb": (400, 640), "left_wrist_0_rgb": (1056, 1280), "right_wrist_0_rgb": (1056, 1280)}


def fake_jpeg(height: int, width: int, seed: int, pattern: str = "smooth") -> bytes:
    from PIL import Image  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    if pattern == "noise":
        # 纯噪声压缩率极低，是上行带宽的最坏情况
        arr = rng.integers(40, 216, size=(height, width, 3), dtype=np.uint8)
    else:
        # 平滑渐变 + 传感器噪声，接近真实相机画面的 JPEG 体积
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        base = np.sin(xx / 40.0 + seed) * 30 + np.cos(yy / 55.0) * 30 + 128
        arr = np.stack([base, base * 0.9 + 15, base * 0.8 + 30], axis=-1)
        arr = np.clip(arr + rng.normal(0, 6, size=arr.shape), 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--prompt", default="pick up the phone")
    p.add_argument("--pattern", choices=("smooth", "noise"), default="smooth",
                   help="smooth=接近真实相机（默认），noise=上行带宽最坏情况")
    args = p.parse_args()

    sock = socket.create_connection((args.host, args.port), timeout=10.0)
    sock.settimeout(60.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"已连接 {args.host}:{args.port}", flush=True)

    state = np.zeros(24, dtype=np.float32)
    times, sizes = [], []

    for i in range(args.rounds):
        blobs = {k: fake_jpeg(h, w, i, args.pattern) for k, (h, w) in SHAPES.items()}
        sizes.append(sum(len(b) for b in blobs.values()))

        t0 = time.monotonic()
        wire.send(sock, {"type": "infer", "prompt": args.prompt}, arrays={"state": state}, blobs=blobs)
        meta, arrays, _ = wire.recv(sock)
        dt = (time.monotonic() - t0) * 1000.0
        times.append(dt)

        if meta.get("type") != "actions":
            print(f"响应异常: {meta}", flush=True)
            return 1
        actions = arrays["actions"]
        if actions.shape != (50, 16):
            print(f"动作形状异常: {actions.shape}", flush=True)
            return 1

        tag = "第 1 次(含编译)" if i == 0 else f"第 {i + 1} 次"
        print(f"{tag:14s} 往返 {dt:7.1f} ms  "
              f"解码 {meta.get('decode_ms', 0):5.1f} + 推理 {meta.get('infer_ms', 0):5.1f} ms  "
              f"上行 {sizes[-1] / 1024:6.1f} KB  "
              f"左臂[:3]={np.array2string(actions[0, :3], precision=3)}", flush=True)

    if len(times) > 1:
        steady = times[1:]
        med = statistics.median(steady)
        net = med - statistics.median([m for m in (meta.get("infer_ms"), meta.get("decode_ms")) if m]) \
            if meta.get("infer_ms") else 0.0
        print(f"\n稳态 中位数 {med:.1f} ms  ({1000 / med:.2f} Hz)", flush=True)
        print(f"其中 服务端推理 {meta.get('infer_ms', 0):.1f} ms + 解码 {meta.get('decode_ms', 0):.1f} ms"
              f" → 网络往返约 {net:.0f} ms", flush=True)
        print(f"上行 {sizes[-1] / 1024 / 1024:.2f} MB/帧（{args.pattern} 模式）", flush=True)
    sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
