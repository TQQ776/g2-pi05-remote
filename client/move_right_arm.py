#!/usr/bin/env python3
"""右臂关节空间运动：把右臂从当前位置缓动到数据集的起始位姿。

目标位姿来自 g2_pick_phone_v21 的 81 个 episode 的**第一帧**取中位数。
实测大部分关节的中位数就等于各自极值，说明每个 episode 都从同一个 home
位姿开始 —— 推理前把右臂摆回这里，首帧动作才落在训练分布内。

    ./run_right_arm.sh                    # 空跑，只打印计划
    ./run_right_arm.sh --run              # 执行，默认 4 秒缓动
    ./run_right_arm.sh --run --duration 8 # 更慢

只动右臂 7 个关节。左臂 7 个目标全程锁定在当前实测值，不会被带着走。
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from g2_io import G2IO

# 数据集 episode 起始帧中位数（81 个 episode），右臂 joint1..7，单位 rad
HOME_RIGHT = np.array([-1.7371, -1.2062, 1.9041, -2.0962, 0.4180, 0.3562, 0.1027])

ARM_DIM = 14
STATE_ARM_SLICE = slice(8, 22)  # state[8:22] = 左臂 7 + 右臂 7
RIGHT = slice(7, 14)

# 单个 tick 允许的最大变化。100 Hz 下 0.02 rad = 2 rad/s，
# 比训练数据里的最大速度（0.38 rad/s）高一个量级，越过它就是参数写错了
MAX_TICK_RAD = 0.02


def cosine_ramp(current: np.ndarray, target: np.ndarray, ticks: int) -> np.ndarray:
    """余弦缓动，起止速度都为零，不会在开始和到位时给电机一个阶跃。"""
    u = np.linspace(0.0, 1.0, ticks)
    s = 0.5 - 0.5 * np.cos(math.pi * u)
    return current[None, :] + s[:, None] * (target - current)[None, :]


def main() -> int:
    p = argparse.ArgumentParser(description="右臂回到数据集起始位姿")
    p.add_argument("--run", action="store_true", help="真正下发指令；不加只打印计划")
    p.add_argument("--duration", type=float, default=4.0, help="缓动总时长（秒）")
    p.add_argument("--hz", type=float, default=100.0, help="伺服下发频率（GDK 要求 100 Hz）")
    p.add_argument("--settle", type=float, default=0.3, help="到位后继续保持多久（秒）")
    p.add_argument("--target", nargs=7, type=float, metavar="J",
                   help="自定义右臂 7 个目标角（rad），不给就用数据集 home")
    args = p.parse_args()

    target_right = (np.asarray(args.target, dtype=np.float64) if args.target
                    else HOME_RIGHT.copy())

    io = G2IO()
    state = io.read_state()  # 缺关节或任何关节报错都会在这里抛
    current = state[STATE_ARM_SLICE].astype(np.float64)

    # 左臂锁定在当前值；右臂去目标
    target = current.copy()
    target[RIGHT] = target_right

    delta = target - current
    ticks = max(2, int(round(args.duration * args.hz)))
    ramp = cosine_ramp(current, target, ticks)
    per_tick = float(np.abs(np.diff(np.vstack([current[None, :], ramp]), axis=0)).max())

    print(f"右臂：当前位置 -> 数据集起始位姿")
    print(f"{args.duration:g} s @ {args.hz:g} Hz = {ticks} 个 tick\n")
    print("关节      当前        目标        差值")
    for j in range(7):
        i = 7 + j
        print(f"  R{j + 1}  {current[i]:+9.4f}  {target[i]:+9.4f}  {delta[i]:+8.4f} rad"
              f"  ({delta[i] * 57.2958:+6.1f}°)")
    print(f"\n左臂 7 个关节锁定在当前值，不参与运动")
    print(f"最大单 tick 变化 {per_tick:.5f} rad（上限 {MAX_TICK_RAD}）")
    print(f"峰值速度约 {per_tick * args.hz:.3f} rad/s")

    if per_tick > MAX_TICK_RAD:
        print("\n拒绝执行：单 tick 变化超过上限，--duration 给得太小了")
        return 3
    if float(np.abs(delta[RIGHT]).max()) < 1e-4:
        print("\n右臂已经在目标位姿，无需动作")
        return 0

    if not args.run:
        print("\n[空跑] 未下发任何指令。确认上面的数值后加 --run 执行。")
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
        return 130

    print("[完成] 复核到位情况：")
    after = io.read_state()[STATE_ARM_SLICE]
    for j in range(7):
        i = 7 + j
        print(f"  R{j + 1}  目标 {target[i]:+9.4f}  实际 {after[i]:+9.4f}"
              f"  误差 {after[i] - target[i]:+.4f}")
    print(f"\n最大到位误差 {float(np.abs(after[RIGHT] - target[RIGHT]).max()):.4f} rad")
    io.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
