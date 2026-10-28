#!/usr/bin/env python3
"""G2 π0.5 远程推理客户端（跑在机器人上）。

流程：
    读三路相机 + 24 维 state  ->  打包发给服务器  ->  收回 (50,16) action
    ->  按固定频率把前 N 步下发给 GDK  ->  重新规划

安全约定：
  * 默认 **不输出任何运动指令**，只打印收到的动作。必须显式加 --run 才会动。
  * 每个动作块先做速度和跳变校验，超限直接拒绝整块，不做截断。
  * 运控故障、关节报错、NaN 一律抛异常退出，不尝试自动恢复。

用法示例：
    python client.py --host 1.2.3.4 --port 8765               # 只读，打印动作
    python client.py --host 1.2.3.4 --port 8765 --run         # 真正控制
"""

from __future__ import annotations

import argparse
import json
import math
import select
import socket
import sys
import threading
import time

import numpy as np

import wire
from g2_io import G2IO

ACTION_DIM = 16
ACTION_HORIZON = 50
ARM_DIM = 14
# 24 维 state 的排布是 [body 5, head 3, 左臂 7, 右臂 7, 夹爪 2]，
# 机械臂 14 维在 8:22 —— 不能直接拿 state[:14] 当锚点，那是躯干和头部。
STATE_ARM_SLICE = slice(8, 22)


class Sender(threading.Thread):
    """后台把打包好的请求写进 socket。

    上行有 300-600 KB，`sendall` 在链路慢的时候会阻塞几百毫秒；
    放到后台线程，主线程就能一边执行当前动作块一边把数据推上去。
    这个线程只碰 socket，**不碰 GDK**，不会和主线程抢机器人。
    """

    def __init__(self, sock: socket.socket) -> None:
        super().__init__(daemon=True, name="g2-sender")
        self.sock = sock
        self._cv = threading.Condition()
        self._payload: bytes | None = None
        self._busy = False
        self._error: BaseException | None = None
        self.last_send_ms = 0.0

    def submit(self, payload: bytes) -> None:
        with self._cv:
            if self._busy:
                raise RuntimeError("上一次发送还没完成")
            self._payload = payload
            self._busy = True
            self._cv.notify()

    def wait(self) -> None:
        with self._cv:
            while self._busy:
                self._cv.wait()
        if self._error is not None:
            raise self._error

    def run(self) -> None:
        while True:
            with self._cv:
                while self._payload is None:
                    self._cv.wait()
                payload, self._payload = self._payload, None
            started = time.monotonic()
            try:
                self.sock.sendall(payload)
            except OSError as exc:  # noqa: BLE001 交给主线程抛
                with self._cv:
                    self._error = exc
                    self._busy = False
                    self._cv.notify_all()
                return
            with self._cv:
                self.last_send_ms = (time.monotonic() - started) * 1000.0
                self._busy = False
                self._cv.notify_all()


def connect(host: str, port: int, retries: int = 30, delay: float = 1.0) -> socket.socket:
    last = None
    for attempt in range(1, retries + 1):
        try:
            sock = socket.create_connection((host, port), timeout=10.0)
            sock.settimeout(30.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError as exc:
            last = exc
            print(f"[连接] 第 {attempt}/{retries} 次失败: {exc}", flush=True)
            time.sleep(delay)
    raise SystemExit(f"无法连接 {host}:{port}: {last}")


def check_chunk(actions: np.ndarray) -> None:
    """校验动作块的形状和数值。"""
    if actions.shape != (ACTION_HORIZON, ACTION_DIM):
        raise RuntimeError(f"动作形状异常: {actions.shape}，期望 {(ACTION_HORIZON, ACTION_DIM)}")
    if not np.isfinite(actions).all():
        raise RuntimeError("动作包含 NaN/Inf")


def check_speed(targets: np.ndarray, anchor: np.ndarray, period: float,
                max_speed: float, max_jump: float) -> None:
    """校验即将下发的这一段手臂目标的速度；不合格拒绝整块，不做截断。

    分两个阈值，因为这两件事的物理来源完全不同（实测，15 次采样）：

    * **块内相邻步**：轨迹本身的运动，最大 0.38 rad/s、中位 0.17。
      这一条是真正的「轨迹是否平滑」，超过 max_speed 说明模型在乱画。
    * **第 0 步**：跨越两块之间的接缝。π0.5 是随机采样的，同一个观测连跑两次
      结果就不一样，接缝差值最大 3.57 rad/s（单步 6.8°）、中位 1.65。
      这是策略固有的采样噪声，不是故障，所以阈值单独放宽到 max_jump。
    """
    if anchor.shape != (ARM_DIM,):
        raise ValueError(
            f"anchor 维度错误: {anchor.shape}，期望 ({ARM_DIM},) "
            "—— 必须是机械臂 14 维，不是整个 24 维 state"
        )
    arm = np.asarray(targets, dtype=np.float64)[:, :ARM_DIM]
    steps = np.diff(np.vstack([anchor[None, :], arm]), axis=0)
    speeds = np.abs(steps) / period

    for row, limit, what in ((0, max_jump, "跨块接缝"), (slice(1, None), max_speed, "块内")):
        seg = speeds[row]
        worst = float(seg.max())
        if worst > limit:
            flat = int(np.argmax(seg))
            # 块内分支的 seg 是二维的，argmax 返回的是展平下标，要还原成关节号
            idx = flat if seg.ndim == 1 else int(np.unravel_index(flat, seg.shape)[1])
            raise RuntimeError(
                f"动作超速（{what}）: 关节 {idx} 需要 {worst:.3f} rad/s "
                f"> 上限 {limit:.3f} rad/s（单步 {worst * period:.4f} rad = "
                f"{np.degrees(worst * period):.1f}°）。请先只读核对预测和当前场景，"
                f"不要盲目调大上限。"
            )


def map_grippers(values: np.ndarray, endpoints: tuple, span: float) -> np.ndarray:
    """把训练量纲的夹爪动作映射到 GDK 目标位置。

    endpoints = (left_open, left_close, right_open, right_close)，
    必须先在真机上标定：模型输出 span 对应张开，0 对应闭合。
    """
    span = float(span)
    if span <= 0:
        raise ValueError("--gripper-span 必须为正")
    frac = np.clip(np.asarray(values, dtype=np.float64) / span, 0.0, 1.0)
    out = np.empty(2, dtype=np.float64)
    for side in (0, 1):
        opened, closed = endpoints[2 * side], endpoints[2 * side + 1]
        out[side] = opened + frac[side] * (closed - opened)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="G2 π0.5 远程推理客户端")
    p.add_argument("--host", required=True, help="服务器 IP")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--prompt", default="pick up the phone", help="任务指令，必须与训练一致")
    p.add_argument("--fps", type=float, default=30.0, help="动作下发频率")
    p.add_argument("--replan-every", type=int, default=10,
                   help="执行多少步后重新推理；越大越省算力，越小越跟手")
    p.add_argument("--max-iters", type=int, default=0, help="推理轮数上限，0 表示不限")
    p.add_argument("--duration", type=float, default=0.0, help="运行秒数上限，0 表示不限")
    p.add_argument("--dump", default="",
                   help="把每轮收到的原始动作块写到这个 jsonl 文件，用来离线分析接缝和平滑度")
    p.add_argument("--jpeg-quality", type=int, default=92,
                   help="相机输出非 JPEG 时的重编码质量")

    cam = p.add_argument_group("相机")
    cam.add_argument("--fresh-timeout", type=float, default=0.05,
                     help="等新帧的总预算（秒，三路合计）。相机约 28.8 Hz，"
                          "50 ms 够跨过一个帧周期；再长只会让主线程多阻塞")
    cam.add_argument("--max-frame-age", type=float, default=1.0,
                     help="画面停滞超过这个秒数就拒绝推理（0 表示不检查）。"
                          "慢节奏采样时实测会拿到重复帧，盲跑很危险")

    motion = p.add_argument_group("运动（全部默认关闭）")
    motion.add_argument("--run", action="store_true", help="真正下发运动指令，不加则只打印")
    motion.add_argument(
        "--max-speed", type=float, default=1.0,
        help="块内相邻步的速度上限 rad/s（实测轨迹最大 0.38，留 2.6 倍余量）",
    )
    motion.add_argument(
        "--max-jump", type=float, default=8.0,
        help="跨块接缝第 0 步的速度上限 rad/s（实测采样噪声最大 3.57，留 2.2 倍余量）",
    )
    motion.add_argument(
        "--servo-hz", type=float, default=100.0,
        help="伺服指令下发频率。GDK 文档要求 move_arm_joint_servo 以 100 Hz 调用；"
             "动作块本身是 --fps 的，会插值到这个频率",
    )
    motion.add_argument(
        "--max-target-rate", type=float, default=2.0,
        help="下发目标的限速率 rad/s。给伺服指令本身限速，接缝跃变会被摊到几十毫秒里，"
             "而不是在一个 10 ms tick 内跳完——后者会让运控报 "
             "target joint error over threshold。正常轨迹只有 0.17–0.38 rad/s，"
             "2.0 不会干扰跟踪；0 表示关闭",
    )
    motion.add_argument(
        "--no-ensemble", action="store_true",
        help="关闭时序集成。默认开启：把最近几块对同一时刻的预测加权平均，"
             "消除分块推理的接缝跳变（实测接缝速度是块内的 13.5 倍）",
    )
    motion.add_argument(
        "--ensemble-decay", type=float, default=0.01,
        help="时序集成的指数衰减系数 m，权重 w=exp(-m·age)。越大越偏向最新一块；"
             "0 表示所有覆盖该时刻的预测等权平均",
    )
    motion.add_argument(
        "--ensemble-taper", type=int, default=12,
        help="动作块末尾这么多步内权重线性衰减到 0。50 步视野的尾巴预测已经退化，"
             "不衰减会在它掉出缓冲时产生块内突变",
    )
    motion.add_argument("--no-speed-limit", action="store_true", help="关闭速度校验（不推荐）")

    grip = p.add_argument_group("夹爪")
    grip.add_argument("--gripper-mode", choices=("hold", "raw", "map"), default="hold",
                      help="hold=不动夹爪；raw=模型输出直接下发；map=按标定端点映射")
    grip.add_argument("--gripper-endpoints", nargs=4, type=float,
                      metavar=("L_OPEN", "L_CLOSE", "R_OPEN", "R_CLOSE"))
    grip.add_argument("--gripper-span", type=float, default=0.025,
                      help="训练数据里夹爪动作的最大值（map 模式的满量程）")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.gripper_mode == "map" and not args.gripper_endpoints:
        print("--gripper-mode map 需要 --gripper-endpoints", file=sys.stderr)
        return 2
    # 时间对齐要丢掉往返期间已经过时的前若干步，必须留够余量
    max_replan = ACTION_HORIZON - 20
    if args.replan_every <= 0 or args.replan_every > max_replan:
        print(f"--replan-every 必须在 1..{max_replan}"
              f"（动作块 {ACTION_HORIZON} 步，要留 20 步给往返延迟的时间对齐）", file=sys.stderr)
        return 2

    period = 1.0 / args.fps
    mode = "控制" if args.run else "只读"
    print(f"[启动] {mode}模式  fps={args.fps:g}  replan_every={args.replan_every}  "
          f"gripper={args.gripper_mode}", flush=True)

    robot = G2IO()
    sock = connect(args.host, args.port)
    sender = Sender(sock)
    sender.start()
    print(f"[连接] {args.host}:{args.port} 已建立", flush=True)

    started = time.monotonic()
    iters = 0
    last_log = 0.0

    def request() -> tuple[np.ndarray, float, float]:
        """采一帧、打包、丢给后台线程发送，**不等结果**。

        返回 (state, t_sent, sensor_ms)。sensor_ms 是采图耗时 ——
        这段是主线程阻塞的，会直接体现为机械臂停顿，所以必须统计出来。
        """
        sensor_started = time.monotonic()
        state = robot.read_state()
        images, _shapes = robot.read_images(args.jpeg_quality, args.fresh_timeout)
        sensor_ms = (time.monotonic() - sensor_started) * 1000.0

        # 帧龄是下界（从「第一次看到这一帧」算起），超限说明画面确实停滞了。
        # 相机实测有 2-4 s 断流，这时按旧画面规划就是盲操作，必须停。
        age = max((v for v in robot.last_frame_age.values() if v is not None), default=0.0)
        if args.max_frame_age > 0 and age > args.max_frame_age:
            fps_text = "、".join(f"{k} {v:.1f}" for k, v in robot.image_fps().items())
            raise RuntimeError(
                f"相机画面已停滞 ≥{age * 1000:.0f} ms，超过 --max-frame-age="
                f"{args.max_frame_age * 1000:.0f} ms。看不到现场就不该继续动，"
                f"先查机器人侧相机（GDK 自报帧率 {fps_text}）。"
            )

        payload = wire.pack({"type": "infer", "prompt": args.prompt},
                            arrays={"state": state}, blobs=images)
        sender.submit(payload)
        return state, time.monotonic(), sensor_ms

    def collect(t_sent: float) -> tuple[np.ndarray, dict, float, int]:
        """收一整块动作，**不裁剪**。

        动作块是「从观测那一刻起，每 period 一步」的计划，所以第 k 步对应
        全局第 obs_step+k 步。时序集成需要保留完整块，让同一时刻能有多块
        的预测可平均；裁剪掉重叠部分正是接缝跳变的来源。

        返回 (完整动作块, meta, 完成时刻, rtt 折算的步数——只用于日志)。
        """
        sender.wait()
        meta, arrays, _ = wire.recv(sock)
        done = time.monotonic()
        if meta.get("type") == "error":
            raise RuntimeError(f"服务器返回错误: {meta.get('message')}")
        if meta.get("type") != "actions":
            raise RuntimeError(f"未知响应类型: {meta.get('type')}")
        actions = arrays["actions"]
        check_chunk(actions)
        skip = int(round((done - t_sent) / period))
        if ACTION_HORIZON - skip < args.replan_every:
            raise RuntimeError(
                f"往返 {skip} 步后只剩 {ACTION_HORIZON - skip} 步，不够 --replan-every="
                f"{args.replan_every}。网络太慢或 replan 太大"
            )
        return actions, meta, done, skip

    # ---- 时序集成缓冲：[(观测时的全局步号, 完整动作块)]，旧的在前 ----
    history: list[tuple[int, np.ndarray]] = []

    def build_plan(start_step: int, n: int) -> np.ndarray:
        """算出全局步 start_step 起的 n 步要执行的目标（已做时序集成）。

        每个时刻把所有覆盖它的预测按 exp(-m·age) 加权平均，age=0 是最新一块。
        实测同一时刻相邻两块的预测中位差 0.084 rad，直接拼接就是每块一次的
        跳变；平均之后接缝被抹平。
        """
        plan = np.zeros((n, ACTION_DIM), dtype=np.float64)
        sources = history[-1:] if args.no_ensemble else history
        taper = max(1, args.ensemble_taper)
        for j in range(n):
            s = start_step + j
            num = np.zeros(ACTION_DIM)
            den = 0.0
            for age, (obs, acts) in enumerate(reversed(sources)):
                k = s - obs
                if 0 <= k < acts.shape[0]:
                    w = math.exp(-args.ensemble_decay * age)
                    # 视野末端（k 接近 50）的预测已经退化，实测同一时刻两种
                    # 预测在这里能差 0.3 rad。让权重在最后 taper 步线性归零，
                    # 否则它掉出缓冲的瞬间均值会突变（实测 4.7 rad/s 的块内跳变）。
                    remain = acts.shape[0] - 1 - k
                    if remain < taper:
                        w *= remain / taper
                    num += w * acts[k]
                    den += w
            if den <= 0.0:
                raise RuntimeError(
                    f"全局第 {s} 步没有任何可用的预测：缓冲里最老的覆盖到 "
                    f"{max((o + a.shape[0] for o, a in history), default=-1)}。"
                    f"说明推理跟不上了，检查网络"
                )
            plan[j] = num / den
        return plan

    try:
        # 第一轮必须同步，否则没有动作可执行
        state, t_sent, _sensor_ms = request()
        first, meta, t_done, skip = collect(t_sent)
        rtt_ms = (t_done - t_sent) * 1000.0
        history.append((0, first.astype(np.float64)))
        print(f"[首帧] rtt={rtt_ms:.0f}ms 推理={meta.get('infer_ms', 0):.0f}ms "
              f"往返折合 {skip} 步，动作块 {first.shape[0]} 步", flush=True)

        # 不打印启动期的 get_image_fps()：GDK 的窗口还没收敛，实测会在 7~140 Hz
        # 之间乱跳，稳定后是 28.8 Hz。相机状况看每轮日志的 `帧龄=` 就够了。

        # 下发限速的起点 = 机械臂当前实际位置。之后每个伺服 tick 都在这个值上累进，
        # 跨块也连续，所以新块第 0 步的采样噪声不会变成一个 10 ms 的阶跃。
        last_target = state[STATE_ARM_SLICE].astype(np.float64)
        step = 0                 # 全局计划步号（30 Hz），history 的时间基准

        while True:
            if args.max_iters and iters >= args.max_iters:
                break
            if args.duration and time.monotonic() - started > args.duration:
                break

            robot.check_faults()

            # ---- 先发下一帧，再执行当前块，最后收结果 ----
            # 发送在后台线程，执行在 333ms 里跑，等收结果时数据通常已经到了，
            # 所以一个往返被一个动作块的执行时间盖住，机械臂不中断。
            _state, t_sent, sensor_ms = request()
            obs_step = step          # 这一帧观测对应的全局步号，收结果时要用

            # 本块执行什么，由**已收到的**预测经时序集成算出。
            # 刚发出去的那一帧要等这块执行完才回来，所以总是落后一块，
            # 这是流水线的固有延迟，集成不改变它，只消除接缝。
            plan = build_plan(step, args.replan_every)
            if not args.no_speed_limit:
                check_speed(plan, last_target, period, args.max_speed, args.max_jump)

            step_start = time.monotonic()
            # 边执行边探测响应是否已经到；这样测到的才是真实网络往返，
            # 而不是「执行完去读 socket」那个被动作块长度撑住的假数。
            arrived = None
            # 动作块是 30 Hz 的（训练数据帧率），但 GDK 的 move_arm_joint_servo
            # 文档要求**以 100 Hz 下发**。所以把每个 33 ms 的计划步线性插值成
            # 若干个 10 ms 的伺服指令；轨迹推进速度不变，只是发得更密。
            tick_period = 1.0 / args.servo_hz
            ticks_per_step = args.servo_hz * period          # 100/30 ≈ 3.33
            n_ticks = max(1, round(args.replan_every * ticks_per_step))
            last_k = -1
            slew = args.max_target_rate * tick_period        # 每个伺服 tick 允许的目标位移
            for i in range(n_ticks):
                u = i / ticks_per_step                      # 处于第几个计划步
                k = min(int(u), args.replan_every - 1)
                frac = min(u - k, 1.0)
                nxt = min(k + 1, args.replan_every - 1)
                target = plan[k, :ARM_DIM] + frac * (plan[nxt, :ARM_DIM] - plan[k, :ARM_DIM])
                # 给指令目标限速。新块第 0 步和上一块末尾的差是策略采样噪声
                # （实测最大 0.27 rad），不限速就是一个 10 ms 内 27 rad/s 的阶跃，
                # 电机跟不动，运控会报 target joint error over threshold 并锁停。
                if slew > 0:
                    target = last_target + np.clip(target - last_target, -slew, slew)
                last_target = target

                if args.run:
                    robot.move_arms(target, tick_period)
                    # 夹爪按计划步下发（30 Hz），不跟着 100 Hz 插值
                    if args.gripper_mode != "hold" and k != last_k:
                        grip = (plan[k, ARM_DIM:]
                                if args.gripper_mode == "raw"
                                else map_grippers(plan[k, ARM_DIM:], tuple(args.gripper_endpoints),
                                                  args.gripper_span))
                        robot.move_grippers(grip)
                    last_k = k

                deadline = step_start + (i + 1) * tick_period
                slack = deadline - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                if arrived is None and select.select([sock], [], [], 0)[0]:
                    arrived = time.monotonic()

            step += args.replan_every

            # 执行完才收结果；正常情况下数据早已到达，这一步几乎不耗时
            wait_start = time.monotonic()
            new_actions, meta, t_done, skip = collect(t_sent)
            stall_ms = (time.monotonic() - wait_start) * 1000.0
            # net_ms 是首个响应字节已到达的时刻（没有则退回到收完的时刻）
            net_ms = ((arrived if arrived is not None else t_done) - t_sent) * 1000.0

            # 整块入缓冲，不做裁剪：它覆盖的全局步号是 obs_step .. obs_step+49
            raw = new_actions.astype(np.float64)
            history.append((obs_step, raw))
            while history and history[0][0] + history[0][1].shape[0] <= step:
                history.pop(0)

            if args.dump:
                with open(args.dump, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "iter": iters,
                        "skip": int(skip),
                        "step": int(step),
                        "sensor_ms": sensor_ms,
                        "net_ms": net_ms,
                        "stall_ms": stall_ms,
                        "history": len(history),
                        "plan": np.round(plan.astype(float), 6).tolist(),
                        "raw": np.round(raw, 6).tolist(),
                    }) + "\n")

            # 每个周期的阻塞 = 采图 + max(0, 网络往返 - 执行预算)；采图本身也会让机械臂停顿
            budget_ms = args.replan_every * period * 1000.0
            overhead_ms = sensor_ms + net_ms
            if overhead_ms > budget_ms:
                need = int(np.ceil(overhead_ms / period / 1000.0)) + 2
                print(f"[警告] 采图 {sensor_ms:.0f} ms + 网络 {net_ms:.0f} ms = {overhead_ms:.0f} ms，"
                      f"超过 {budget_ms:.0f} ms 的执行预算，本轮停顿 {sensor_ms + stall_ms:.0f} ms。"
                      f"把 --replan-every 提到 {need} 以上可消除停顿。", flush=True)
            elif overhead_ms > budget_ms * 0.8 and iters % 30 == 0:
                print(f"[提示] 采图 {sensor_ms:.0f} + 网络 {net_ms:.0f} = {overhead_ms:.0f} ms / "
                      f"预算 {budget_ms:.0f} ms，余量不足 20%", flush=True)

            iters += 1
            now = time.monotonic()
            if now - last_log >= 1.0:
                last_log = now
                a0 = plan[0, :ARM_DIM]
                ages = [v for v in robot.last_frame_age.values() if v is not None]
                age_ms = max(ages) * 1000.0 if ages else 0.0
                print(f"[{iters:4d}] 网络={net_ms:6.1f}ms  推理={meta.get('infer_ms', 0):6.1f}ms  "
                      f"对齐跳过={skip:2d}步  块间停顿={sensor_ms + stall_ms:5.1f}ms  "
                      f"帧龄={age_ms:5.0f}ms  "
                      f"左臂={np.array2string(a0[:3], precision=3)}  "
                      f"夹爪=({plan[0, 14]:+.4f}, {plan[0, 15]:+.4f})", flush=True)

    except KeyboardInterrupt:
        print("\n[退出] 收到中断", flush=True)
        return 130
    finally:
        sock.close()

    clip = getattr(robot, "_gripper_clip_count", 0)
    extra = f"，夹爪目标钳位 {clip} 次（模型输出略超 [0, 0.025]）" if clip else ""
    print(f"[结束] 共 {iters} 轮推理{extra}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
