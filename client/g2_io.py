"""G2 机器人的 GDK 读写封装。

只在机器人本体上可用：agibot_gdk 是 aarch64 / CPython 3.10 的编译扩展，
在 x86_64 开发机上 import 会失败。所以本文件把 `import agibot_gdk` 放在
__init__ 里，模块本身可以被任何机器 import（单元测试、--help 都不会初始化 DDS）。
（混合部署模式下 x86_64 也有一份 GDK，见 gdk/installer/server_installer.tar.gz。）

state 的 24 维顺序必须和训练数据严格一致（见数据集 meta/info.json）：
    [body 5, head 3, 左臂 7, 右臂 7, 左夹爪, 右夹爪]
action 的 16 维：
    [左臂 7, 右臂 7, 左夹爪, 右夹爪]

夹爪数值直接沿用训练数据的原始量纲（0 ~ 0.025 量级），不做单位换算；
是否需要映射到 GDK 的关节弧度由调用方通过 --gripper-mode 决定。
"""

from __future__ import annotations

import io
import time

import numpy as np

# 顺序 = 训练数据 observation.state 的前 22 维
STATE_JOINT_NAMES = (
    [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
    + [f"idx1{i}_head_joint{i}" for i in range(1, 4)]
    + [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
    + [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
)
ARM_NAMES = STATE_JOINT_NAMES[8:22]

# dual_tool 夹爪位置的有效范围（弧度）。GDK 的校验报错原文：
# "end_effector joint[0] position out of range [0, 0.025]"。反馈侧读回来的
# 也是这个量纲，所以模型输出可以直接用。
GRIPPER_MIN = 0.0
GRIPPER_MAX = 0.025

# 相机键名 -> 训练数据里的 feature 名
CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

# 关节 error_code 的含义，抄自 GDK 离线文档
# gdk/site/contents/appendices/joint_error_code/
# 躯干/头部和手臂用的是两套码表，所以按关节前缀分表。
BODY_HEAD_ERRORS = {
    0x01: "FOC错误", 0x02: "过压", 0x04: "欠压", 0x08: "过温(>110°C)",
    0x10: "启动失败", 0x20: "编码器错误", 0x40: "过流/过载", 0x80: "软件错误",
    0x100: "温度传感器错误", 0x200: "位置超限", 0x400: "关节ID非法",
    0x800: "位置跟踪错误", 0x1000: "电流检测错误", 0x2000: "抱闸打开失败",
    0x4000: "位置指令阶跃警告", 0x8000: "多圈关节丢圈数", 0xF000: "通信丢帧",
    0x10000: "堵转", 0x20000: "超速", 0x40000: "主机丢失(心跳掉线)",
    0x80000: "速度误差过大",
}
ARM_ERRORS = {
    0x2250: "驱动器短路", 0x2310: "U相输出电流过大", 0x2311: "V相输出电流过大",
    0x2312: "W相输出电流过大", 0x2320: "驱动器硬件过流", 0x2330: "驱动器输出对地短路",
    0x3130: "主电源输入异常", 0x3210: "直流母线过压", 0x3220: "直流母线欠压",
    0x4210: "功率模块过热", 0x6010: "CPU1看门狗溢出", 0x6011: "CPU2看门狗溢出",
    0x7112: "能耗制动电阻过载", 0x8311: "电机持续过载", 0x8611: "位置跟随误差过大",
    0x8612: "正向软限位", 0x8613: "负向软限位", 0x8800: "第一编码器数据溢出",
    0x8801: "第二编码器数据溢出", 0xFF00: "CPU1工作异常", 0xFF01: "CPU2工作异常",
    0xFF02: "CPU1内存异常", 0xFF03: "CPU2内存异常", 0xFF04: "CPU内存冲突",
    0xFF05: "磁极定位错误", 0xFF06: "第一编码器数据异常", 0xFF07: "第一编码器通信异常",
    0xFF08: "第一编码器通信超时", 0xFF09: "第一编码器内部异常",
    0xFF10: "第二编码器数据异常", 0xFF11: "第二编码器通信异常",
    0xFF12: "第二编码器通信超时", 0xFF13: "第二编码器内部异常",
}


def describe_joint_error(name: str, code: int) -> str:
    """把关节错误码翻译成人话。查不到就原样返回码值。

    躯干/头部那套码是位标志，可能是多位的组合；手臂那套是整值码表。
    """
    if "_arm_" in name:
        return ARM_ERRORS.get(code, "未在 GDK 文档中的错误码")
    if code in BODY_HEAD_ERRORS:
        return BODY_HEAD_ERRORS[code]
    bits = [text for bit, text in BODY_HEAD_ERRORS.items() if code & bit]
    return " + ".join(bits) if bits else "未在 GDK 文档中的错误码"


class G2IO:
    """封装相机取图、关节读取、机械臂伺服和夹爪控制。"""

    def __init__(self, *, init_timeout: float = 2.0):
        import agibot_gdk as gdk  # noqa: PLC0415  只在真机上可用

        self._gdk = gdk
        if gdk.gdk_init() != gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        self.robot = gdk.Robot()
        self.camera = gdk.Camera()
        self._kinds = (
            gdk.CameraType.kHeadColor,
            gdk.CameraType.kHandLeftColor,
            gdk.CameraType.kHandRightColor,
        )
        self._last_ts: dict = {}    # 每路相机上一帧的 timestamp_ns，用来识别重复帧
        self._last_seen: dict = {}  # 该帧第一次被观察到的主机时刻，用来估算帧龄
        self.last_frame_age: dict = {}  # {相机键: 秒}，read_images 之后可读
        self._gripper_clip_count = 0    # 夹爪目标被钳位的次数，收尾时可以打出来
        # 相机/关节服务需要一点时间才能给出第一帧
        time.sleep(init_timeout)

    # ------------------------------------------------------------------ 读
    def read_state(self) -> np.ndarray:
        """返回 24 维 float32 state。任何缺帧、故障或 NaN 都直接抛异常。"""
        report = self.robot.get_joint_states()
        by_name = {s["name"]: s for s in report["states"]}

        joints = []
        for name in STATE_JOINT_NAMES:
            entry = by_name.get(name)
            if entry is None:
                raise RuntimeError(
                    f"关节反馈缺少 {name}（本次只发布 {len(by_name)} 个关节："
                    f"{', '.join(by_name)}）。电源关掉的关节不会出现在反馈里，"
                    f"先查 get_chest_power_state() 的 *_motor_power_state。"
                )
            code = entry.get("error_code", 0)
            if code:
                raise RuntimeError(
                    f"关节 {name} 报错 0x{code:X}（{code}）= "
                    f"{describe_joint_error(name, code)}。{self._power_hint()}"
                )
            # 文档明确说 position/velocity 是低速电机预留字段，机械臂要看 motor_*
            joints.append(float(entry.get("motor_position", entry["position"])))

        ends = self.robot.get_end_state()
        grippers = []
        for side in ("left", "right"):
            states = ends[f"{side}_end_state"]["end_states"]
            if not states or states[0].get("err_code", 0):
                raise RuntimeError(f"{side} 夹爪反馈缺失或故障")
            grippers.append(float(states[0]["position"]))

        state = np.asarray(joints + grippers, dtype=np.float32)
        if state.shape != (24,):
            raise RuntimeError(f"state 维度错误: {state.shape}")
        if not np.isfinite(state).all():
            raise RuntimeError("state 包含 NaN/Inf")
        return state

    def read_images(self, jpeg_quality: int = 92, fresh_timeout: float = 0.05) -> tuple[dict, dict]:
        """返回 ({相机键: JPEG 字节}, {相机键: 图像尺寸})。

        GDK 本身就出 JPEG 时直接转发原始字节，不重新编码，避免多一次有损压缩。

        `get_latest_image` 在两次推送之间会**重复返回同一帧**，直接用会把静止画面喂给模型，
        所以这里用 timestamp_ns 确认是新帧。

        **相机实测 ~28.8 Hz**（三路轮流取图，设备时间戳中位间隔 33.3 ms，
        270/287 落在 33.3 ms 网格上），和训练数据的 30 fps 一致。
        但**紧轮询会把帧率打下悬崖**：同一台相机 2 ms 轮询只收到 2.4 Hz，
        500 ms 间隔取图则每次都拿到新帧、耗时 <0.1 ms —— 高频调用会饿死
        GDK 内部接收线程。所以重试间隔固定 20 ms，不做紧轮询。

        `fresh_timeout` 是**三路合计**的预算，不是每路的：相机串行等三次，
        每路各等 150 ms 会让主线程整整停 450 ms。
        """
        from PIL import Image  # noqa: PLC0415

        blobs: dict = {}
        shapes: dict = {}
        deadline = time.monotonic() + fresh_timeout

        for key, kind in zip(CAMERA_KEYS, self._kinds):
            image = self._wait_fresh(key, kind, deadline)
            data = bytes(image.data)
            gdk = self._gdk

            if image.encoding in (gdk.Encoding.JPEG, gdk.Encoding.PNG):
                blobs[key] = data
                shapes[key] = (int(image.height), int(image.width))
                continue

            if image.encoding != gdk.Encoding.UNCOMPRESSED:
                raise RuntimeError(f"不支持的图像编码: {image.encoding}")

            arr = np.frombuffer(data, np.uint8).reshape(image.height, image.width, 3)
            if image.color_format == gdk.ColorFormat.BGR:
                arr = arr[:, :, ::-1]
            elif image.color_format != gdk.ColorFormat.RGB:
                raise RuntimeError(f"不支持的像素格式: {image.color_format}")

            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="JPEG", quality=jpeg_quality)
            blobs[key] = buf.getvalue()
            shapes[key] = (int(image.height), int(image.width))

        return blobs, shapes

    def _wait_fresh(self, key: str, kind, deadline: float):
        """取一帧时间戳比上次新的图；到 deadline 还没新的就退回当前缓存帧。

        deadline 是**绝对时刻**（time.monotonic），由调用方在三路相机之间共享，
        保证整段 `read_images` 的阻塞时间不超过一个预算。
        """
        last_ts = self._last_ts.get(key)
        while True:
            # 每次取图也不能超出剩余预算，否则单次调用就能把 deadline 冲掉。
            # 下限给 1 ms —— 传 0 的语义未经验证（可能是「无限等」）。
            budget_ms = max(1.0, min(50.0, (deadline - time.monotonic()) * 1000.0))
            try:
                image = self.camera.get_latest_image(kind, budget_ms)
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"相机 {kind} 没有图像") from None
                time.sleep(0.005)
                continue
            ts = getattr(image, "timestamp_ns", None)
            now = time.monotonic()
            if last_ts is None or ts != last_ts or now >= deadline:
                if last_ts is None or ts != last_ts:
                    self._last_ts[key] = ts
                    self._last_seen[key] = now
                # 帧龄是从「第一次看到这一帧」算起的下界，不是真实拍摄时刻的差
                seen = self._last_seen.get(key)
                self.last_frame_age[key] = None if seen is None else now - seen
                return image
            # 重试间隔必须 > 一帧周期。用 2 ms 紧轮询会饿死 GDK 的接收线程：
            # 实测同一台相机，2 ms 轮询只收到 2.4 Hz，500 ms 轮询能收到 28.8 Hz。
            time.sleep(0.02)

    def power_state(self) -> dict:
        """胸部电源板里各电机组的供电状态。1=有电，0=没电。

        手臂/腰部电机没供电时，驱动器会报 0x3220「直流母线欠压」并被锁停，
        关节也不会出现在 get_joint_states() 里 —— 这是硬件状态，不是软件能绕过的。
        """
        st = self.robot.get_chest_power_state()
        d = st if isinstance(st, dict) else {
            k: getattr(st, k) for k in dir(st) if not k.startswith("_")
        }
        keys = [k for k in d if k.endswith("_power_state") or k.endswith("_brake_enable_state")]
        return {k: d[k] for k in keys if not callable(d[k])}

    def _power_hint(self) -> str:
        """把电源状态压成一句话，附在关节报错后面。"""
        try:
            ps = self.power_state()
        except Exception:  # noqa: BLE001  诊断信息读不到不该掩盖原始错误
            return ""
        off = sorted(k for k, v in ps.items() if v == 0)
        if not off:
            return "各电机组供电正常，故障可能需要 clear_hal_error() 复位。"
        return (f"以下电源为 0（未供电）：{', '.join(off)}。"
                f"先给对应部位上电，再用 clear_hal_error() 复位锁停的电机。")

    def image_fps(self) -> dict:
        """三路相机 GDK 自报的帧率（约 28-30 Hz，和训练数据一致）。

        这个值是**调用方式相关**的：紧轮询时会掉到 2-6 Hz。看到异常低的值，
        先怀疑轮询太密，不要直接下结论说相机坏了。
        刚订阅上时 GDK 还没攒够窗口，会返回 0.0 —— 0 不等于「不出帧」。
        """
        return {key: self.camera.get_image_fps(kind)
                for key, kind in zip(CAMERA_KEYS, self._kinds)}

    def close(self) -> None:
        """释放相机订阅和 DDS。

        不调的话 GDK 的后台线程会吊着进程：实测关掉 tkinter 窗口后进程还在，
        Ctrl-C 也杀不掉（SIGINT 被当成 Tk 回调里的异常吞了）。
        """
        try:
            self.camera.close_camera()
        except Exception:  # noqa: BLE001  退出路径上的清理失败不值得再抛
            pass
        try:
            self._gdk.gdk_release()
        except Exception:  # noqa: BLE001
            pass

    def check_faults(self) -> None:
        status = self.robot.get_motion_control_status()
        if status.error_code != 0:
            raise RuntimeError(
                f"运控故障 error_code={status.error_code}, mode={status.mode}, "
                f"control_mode={status.control_mode}: {status.error_msg}"
            )

    # ------------------------------------------------------------------ 写
    def move_arms(self, target: np.ndarray, period: float) -> None:
        """下发一次 14 维机械臂伺服目标。period 是调用间隔（秒）。"""
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (14,):
            raise ValueError(f"机械臂目标维度错误: {target.shape}")
        if not np.isfinite(target).all():
            raise ValueError("机械臂目标包含 NaN/Inf")
        result = self.robot.move_arm_joint_servo(target.tolist(), float(period), 2, False)
        if result not in (None, 0):
            raise RuntimeError(f"GDK 机械臂伺服发送失败: {result}")

    def move_grippers(self, values: np.ndarray) -> None:
        """下发左右夹爪目标（已经是 GDK 可接受的量纲）。

        GDK 对 dual_tool 的夹爪位置校验范围是 [0, 0.025]（弧度），实测
        move_end_effector_joint 接受的就是 get_end_state 读回来的同一个量纲，
        所以模型输出可以直接下发，不需要映射。模型/时序集成的输出会偶尔
        溢出边界千分之一（实测 0.0250106），GDK 会整条命令判失败，所以这里
        先钳位——超出的部分物理上也走不到。
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (2,):
            raise ValueError(f"夹爪目标维度错误: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("夹爪目标包含 NaN/Inf")
        clipped = np.clip(values, GRIPPER_MIN, GRIPPER_MAX)
        if not np.allclose(clipped, values):
            self._gripper_clip_count += 1
        result = self.robot.move_end_effector_joint(clipped.tolist(), [0.0, 0.0], "dual_tool")
        if result not in (None, 0):
            raise RuntimeError(f"GDK 夹爪伺服发送失败: {result}")
