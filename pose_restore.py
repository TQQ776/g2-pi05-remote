#!/usr/bin/env python3
"""从 pose_save.py 保存的 JSON 还原机械臂姿态。

默认还原左右两臂共 14 个关节，躯干和头部不动（那几个电机当前未上电，
而且还原它们需要 move_waist_joint / move_head_joint，不在本脚本范围内）。

缓动是余弦曲线，起止速度为零，不会在开始和到位时给电机阶跃。
默认只打印计划，加 --run 才真的下发。

    ./run_pose_restore.sh                     # 空跑，看计划
    ./run_pose_restore.sh --run               # 执行
    ./run_pose_restore.sh --run --side right  # 只还原右臂
    ./run_pose_restore.sh --file poses/xxx.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from g2_io import G2IO

POSE_DIR = Path(__file__).resolve().parent / "poses"

# 14 臂关节在 24 维 state 里的位置
LEFT = slice(0, 7)    # 相对 14 维
RIGHT = slice(7, 14)
STATE_ARM_SLICE = slice(8, 22)

MAX_TICK_RAD = 0.02   # 单 tick 上限；100 Hz 下 0.02 rad = 2 rad/s
CRUISE_SPEED = 0.30   # 自动算时长时用的峰值速度（训练数据里左臂最大 0.38）


def cosine_ramp(current: np.ndarray, target: np.ndarray, ticks: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, ticks)
    s = 0.5 - 0.5 * np.cos(math.pi * u)
    return current[None, :] + s[:, None] * (target - current)[None, :]


def main() -> int:
    p = argparse.ArgumentParser(description="还原机械臂姿态")
    p.add_argument("--file", default=str(POSE_DIR / "initial.json"), help="pose_save.py 输出的 JSON")
    p.add_argument("--run", action="store_true", help="真正下发；不加只打印计划")
    p.add_argument("--side", choices=("both", "left", "right"), default="both")
    p.add_argument("--duration", type=float, default=None,
                   help="缓动总时长（秒）；不给则按峰值 0.30 rad/s 自动算")
    p.add_argument("--hz", type=float, default=100.0)
    p.add_argument("--settle", type=float, default=0.3, help="到位后继续保持多久（秒）")
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"找不到姿态文件: {path}")
        return 2
    pose = json.loads(path.read_text(encoding="utf-8"))
    saved = np.asarray(pose["state24"], dtype=np.float64)
    if saved.shape != (24,):
        print(f"姿态文件 state24 维度错误: {saved.shape}")
        return 2

    io = G2IO()
    current = io.read_state()[STATE_ARM_SLICE].astype(np.float64)
    target_right = saved[STATE_ARM_SLICE].astype(np.float64)

    target = current.copy()
    active = {"both": (slice(0, 14)), "left": LEFT, "right": RIGHT}[args.side]
    target[active] = target_right[active]

    delta = target - current
    max_delta = float(np.abs(delta[active]).max())
    duration = args.duration
    if duration is None:
        duration = max(2.0, (math.pi / 2) * max_delta / CRUISE_SPEED)
    ticks = max(2, int(round(duration * args.hz)))
    ramp = cosine_ramp(current, target, ticks)
    per_tick = float(np.abs(np.diff(np.vstack([current[None, :], ramp]), axis=0)).max())

    print(f"姿态文件 {path.name}  采集于 {pose.get('saved_at', '?')}")
    print(f"还原 {args.side}，{duration:.1f} s @ {args.hz:g} Hz = {ticks} 个 tick\n")
    print("关节      当前        目标        差值")
    for j in range(14):
        i = j
        tag = "" if delta[i] != 0 else "  (不动)"
        side = "L" if i < 7 else "R"
        print(f"  {side}{i % 7 + 1}  {current[i]:+9.4f}  {target[i]:+9.4f}"
              f"  {delta[i]:+8.4f} rad ({delta[i] * 57.2958:+6.1f}°){tag}")
    print(f"\n最大单关节位移 {max_delta:.4f} rad")
    print(f"最大单 tick 变化 {per_tick:.5f} rad（上限 {MAX_TICK_RAD}）")
    print(f"峰值速度约 {per_tick * args.hz:.3f} rad/s")

    if per_tick > MAX_TICK_RAD:
        print("\n拒绝执行：单 tick 变化超过上限，把 --duration 调大")
        io.close()
        return 3
    if max_delta < 1e-4:
        print("\n已经在目标姿态，无需动作")
        io.close()
        return 0
    if not args.run:
        print("\n[空跑] 未下发任何指令。确认后加 --run 执行。")
        io.close()
        return 0

    period = 1.0 / args.hz
    print("\n[执行] Ctrl-C 中断后机械臂停在当前位置")
    started = time.monotonic()
    try:
        for i, tgt in enumerate(ramp):
            io.move_arms(tgt, period)
            slack = started + (i + 1) * period - time.monotonic()
            if slack > 0:
                time.sleep(slack)
        hold = time.monotonic() + args.settle
        while time.monotonic() < hold:
            io.move_arms(target, period)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[中断] 已停止下发")
        io.close()
        return 130

    after = io.read_state()[STATE_ARM_SLICE]
    print("[完成] 复核：")
    for j in range(14):
        side = "L" if j < 7 else "R"
        print(f"  {side}{j % 7 + 1}  目标 {target[j]:+9.4f}  实际 {after[j]:+9.4f}"
              f"  误差 {after[j] - target[j]:+.4f}")
    print(f"\n最大到位误差 {float(np.abs(after[active] - target[active]).max()):.4f} rad")
    io.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
