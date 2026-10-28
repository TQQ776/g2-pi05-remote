#!/usr/bin/env python3
"""读取当前机器人姿态并存成 JSON，供 pose_restore.py 还原。

**只读**，不发任何运动指令。

存的是完整 24 维 state（躯干 5 + 头部 3 + 双臂 14 + 夹爪 2），外加：
  - 每个关节的 position / motor_position / velocity / error_code
  - 末端执行器的 type / names（决定夹爪量纲的解释方式）
  - 电源状态、运控状态
再存一份按名字索引的原始转速，方便以后核对 GDK 有没有改字段。

    ./run_pose_save.sh                      # poses/pose_<时间戳>.json
    ./run_pose_save.sh --name initial       # poses/initial.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from g2_io import STATE_JOINT_NAMES, G2IO

POSE_DIR = Path(__file__).resolve().parent / "poses"


def main() -> int:
    p = argparse.ArgumentParser(description="保存当前机器人姿态")
    p.add_argument("--name", default="", help="文件名（不含扩展名）；不给就用时间戳")
    p.add_argument("--note", default="", help="备注，写进文件里")
    args = p.parse_args()

    io = G2IO()
    report = io.robot.get_joint_states()
    by_name = {s["name"]: s for s in report["states"]}
    missing = [n for n in STATE_JOINT_NAMES if n not in by_name]
    if missing:
        raise RuntimeError(f"缺少关节反馈: {', '.join(missing)}（共发布 {len(by_name)} 个）")

    joints = {}
    vector = []
    for name in STATE_JOINT_NAMES:
        s = by_name[name]
        pos = float(s.get("motor_position", s["position"]))
        vector.append(pos)
        joints[name] = {
            "position": float(s["position"]),
            "motor_position": float(s.get("motor_position", s["position"])),
            "velocity": float(s.get("motor_velocity", s.get("velocity", 0.0))),
            "mode": int(s.get("mode", 0)),
            "error_code": int(s.get("error_code", 0)),
        }

    ends = io.robot.get_end_state()
    grippers = []
    end_info = {}
    for side in ("left", "right"):
        block = ends[f"{side}_end_state"]
        states = block["end_states"]
        if not states:
            raise RuntimeError(f"{side} 夹爪无反馈")
        e0 = states[0]
        grippers.append(float(e0["position"]))
        end_info[side] = {
            "type": block.get("type"),
            "names": block.get("names"),
            "controlled": block.get("controlled"),
            "position": float(e0["position"]),
            "velocity": float(e0.get("velocity", 0.0)),
            "err_code": int(e0.get("err_code", 0)),
            "enable": bool(e0.get("enable", False)),
        }

    state24 = np.asarray(vector + grippers, dtype=np.float64)
    assert state24.shape == (24,), state24.shape

    doc = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "note": args.note,
        "joint_timestamp_ns": report.get("timestamp"),
        "state24": [round(float(v), 8) for v in state24],
        "joints": joints,
        "grippers": dict(zip(("left", "right"), grippers)),
        "end_effector": end_info,
        "power_state": io.power_state(),
    }
    try:
        st = io.robot.get_motion_control_status()
        doc["motion_control"] = {
            "error_code": int(getattr(st, "error_code", -1)),
            "mode": getattr(st, "mode", None),
            "control_mode": getattr(st, "control_mode", None),
        }
    except Exception as exc:  # noqa: BLE001  运控状态读不到不该影响姿态保存
        doc["motion_control"] = {"error": str(exc)}

    POSE_DIR.mkdir(exist_ok=True)
    name = args.name or f"pose_{datetime.now():%Y%m%d_%H%M%S}"
    path = POSE_DIR / f"{name}.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"已保存 {path}")
    print(f"采集时刻 {doc['saved_at']}   关节时间戳 {doc['joint_timestamp_ns']}")
    print("\n24 维 state：")
    labels = ([f"body{i}" for i in range(1, 6)] + [f"head{i}" for i in range(1, 4)]
              + [f"L{i}" for i in range(1, 8)] + [f"R{i}" for i in range(1, 8)]
              + ["gripL", "gripR"])
    for i, (lab, val) in enumerate(zip(labels, state24)):
        print(f"  [{i:2d}] {lab:7s} {val:+9.4f}")
    print("\n末端执行器：")
    for side, info in end_info.items():
        print(f"  {side:5s} type={info['type']}  names={info['names']}  "
              f"position={info['position']:+.4f}  err={info['err_code']}")
    io.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
