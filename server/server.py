#!/usr/bin/env python3
"""G2 π0.5 远程推理服务。

接收机器人端发来的三路相机 JPEG + 24 维 state，跑 openpi 策略，返回 (50, 16) action。

协议见同目录 wire.py：4 字节长度 + JSON 头 + 原始字节块。
单客户端串行处理 —— JAX 推理本身不保证多线程并发安全，多客户端排队即可。

    ./run_server.sh                       # 默认 0.0.0.0:8765
    python server.py --port 9000 --no-warmup
"""

from __future__ import annotations

import argparse
import io
import os
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # openpi/

# 必须在 import jax 之前设置
os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / ".openpi_cache"))
os.environ.setdefault("HF_LEROBOT_HOME", "/root/autodl-tmp/hf_cache/lerobot")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.97")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import wire  # noqa: E402

CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
STATE_DIM = 24
ACTION_HORIZON = 50
ACTION_DIM = 16

POLICY = None
LOCK = threading.Lock()
DEFAULT_PROMPT = "pick up the phone"


def decode_jpeg(blob: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(blob)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        conn = self.request
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(120.0)
        peer = self.client_address
        print(f"[连接] {peer[0]}:{peer[1]} 已接入", flush=True)
        frames = 0

        try:
            while True:
                meta, arrays, blobs = wire.recv(conn)
                if meta.get("type") != "infer":
                    wire.send(conn, {"type": "error", "message": f"未知请求类型 {meta.get('type')}"})
                    continue

                try:
                    state = np.asarray(arrays["state"], dtype=np.float32)
                    if state.shape != (STATE_DIM,):
                        raise ValueError(f"state 维度 {state.shape}，期望 ({STATE_DIM},)")
                    if not np.isfinite(state).all():
                        raise ValueError("state 含 NaN/Inf")

                    missing = [k for k in CAMERA_KEYS if k not in blobs]
                    if missing:
                        raise ValueError(f"缺少相机 {missing}")
                    decode_started = time.perf_counter()
                    images = {k: decode_jpeg(blobs[k]) for k in CAMERA_KEYS}
                    decode_ms = (time.perf_counter() - decode_started) * 1000.0

                    obs = {
                        "state": state,
                        "image": images,
                        "prompt": meta.get("prompt") or DEFAULT_PROMPT,
                    }

                    started = time.perf_counter()
                    with LOCK:
                        out = POLICY.infer(obs)
                    infer_ms = (time.perf_counter() - started) * 1000.0

                    actions = np.asarray(out["actions"], dtype=np.float32)
                    if actions.shape != (ACTION_HORIZON, ACTION_DIM):
                        raise ValueError(f"动作形状 {actions.shape}，期望 {(ACTION_HORIZON, ACTION_DIM)}")

                    wire.send(conn, {"type": "actions", "infer_ms": infer_ms, "decode_ms": decode_ms},
                              arrays={"actions": actions})
                    frames += 1
                    if frames % 10 == 1:
                        print(f"[推理] 第 {frames} 帧 {infer_ms:.1f} ms  解码 {decode_ms:.1f} ms  "
                              f"图尺寸={ {k: v.shape[:2] for k, v in images.items()} }", flush=True)

                except Exception as exc:  # noqa: BLE001 单个请求出错不该断开连接
                    print(f"[错误] {type(exc).__name__}: {exc}", flush=True)
                    try:
                        wire.send(conn, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                    except OSError:
                        break

        except (ConnectionError, socket.timeout, OSError) as exc:
            print(f"[断开] {peer[0]}:{peer[1]} {type(exc).__name__}: {exc}  共 {frames} 帧", flush=True)
        finally:
            conn.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    global POLICY, DEFAULT_PROMPT

    parser = argparse.ArgumentParser(description="G2 π0.5 远程推理服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default="pi05_g2_lora", help="openpi 训练配置名")
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints/pi05_g2_lora/g2_pick_phone_v21/19999"),
        help="checkpoint 目录",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="客户端没给 prompt 时的默认值")
    parser.add_argument("--no-warmup", action="store_true", help="跳过启动预热")
    args = parser.parse_args()

    DEFAULT_PROMPT = args.prompt

    if not Path(args.ckpt).is_dir():
        print(f"checkpoint 不存在: {args.ckpt}", file=sys.stderr)
        return 2

    from openpi.policies import policy_config
    from openpi.training import config as _config

    print(f"[加载] {args.config} <- {args.ckpt}", flush=True)
    started = time.time()
    train_config = _config.get_config(args.config)
    POLICY = policy_config.create_trained_policy(train_config, args.ckpt)
    print(f"[加载] 完成，用时 {time.time() - started:.1f}s", flush=True)

    if not args.no_warmup:
        from openpi.policies.g2_policy import make_g2_example

        print("[预热] 首次编译 JAX 图，通常 30-60s ...", flush=True)
        started = time.time()
        POLICY.infer(make_g2_example())
        print(f"[预热] 完成，用时 {time.time() - started:.1f}s", flush=True)

    with Server((args.host, args.port), Handler) as server:
        print(f"[就绪] 监听 {args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[退出] 收到中断", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
