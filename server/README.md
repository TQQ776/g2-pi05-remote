# G2 π0.5 远程推理服务

接收机器人端发来的三路相机 JPEG + 24 维 state，跑 openpi 策略，返回 `(50, 16)` 的 action chunk。

## 启动

```bash
cd /root/autodl-tmp/openpi
./server/run_server.sh                     # 默认 0.0.0.0:8765
./server/run_server.sh --port 9000        # 换端口
./server/run_server.sh --no-warmup        # 跳过预热，首个请求会慢 ~40s
```

启动日志正常长这样：

```
[加载] pi05_g2_lora <- /root/autodl-tmp/openpi/checkpoints/pi05_g2_lora/g2_pick_phone_v21/19999
[加载] 完成，用时 8.0s
[预热] 首次编译 JAX 图，通常 30-60s ...
[预热] 完成，用时 38.9s
[就绪] 监听 0.0.0.0:8765
[连接] 1.2.3.4:52311 已接入
[推理] 第 1 帧 129.0 ms  图尺寸={'base_0_rgb': (400, 640), ...}
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8765` | 监听端口 |
| `--config` | `pi05_g2_lora` | openpi 配置名 |
| `--ckpt` | `checkpoints/pi05_g2_lora/g2_pick_phone_v21/19999` | checkpoint 目录 |
| `--prompt` | `pick up the phone` | 客户端未提供时的默认指令 |
| `--no-warmup` | 关 | 跳过启动时的编译预热 |

## 端口放行

AutoDL 容器需要在控制台的「自定义服务」里把端口映射出去，或者用 SSH 隧道：

```bash
# 在本地机器人那台机器上
ssh -N -L 8765:127.0.0.1:8765 root@<服务器地址> -p <端口>
# 然后客户端连 127.0.0.1:8765
```

## 协议

`wire.py`：4 字节大端长度 + JSON 头 + 原始字节块，只依赖标准库和 numpy。

**请求**
```
meta  = {"type": "infer", "prompt": "pick up the phone"}
arrays = {"state": float32[24]}                    # [body5, head3, 左臂7, 右臂7, 夹爪2]
blobs  = {"base_0_rgb": jpeg, "left_wrist_0_rgb": jpeg, "right_wrist_0_rgb": jpeg}
```

**响应**
```
meta   = {"type": "actions", "infer_ms": 116.6, "decode_ms": 42.1}
arrays = {"actions": float32[50, 16]}              # [左臂7, 右臂7, 夹爪2]
```

出错时返回 `{"type": "error", "message": "..."}`，连接保持不断。

## 实测性能（RTX 4090）

| 项 | 数值 |
|---|---|
| 模型加载 | 7.6 s |
| 首次编译（预热） | 40.7 s |
| 三路 JPEG 解码 | 42 ms |
| 单次推理 | **117 ms（8.5 Hz）** |
| 服务端合计 | 159 ms |
| 显存 | 6.34 GiB |

推理的 117 ms 里：预处理 19 ms、模型前向 85 ms、输出反归一化约 25 ms。

端到端往返还要加上行传输时间，客户端那边算。两路腕部相机是 1056×1280，
拟真画面约 285 KB/路、头部 54 KB，合计 626 KB/帧。

## 注意

- 服务端**不做**任何动作安全校验，速度限制、跳变检查都在客户端。
- 解码和推理都计入返回的 `decode_ms` / `infer_ms`，客户端据此判断会不会停顿。
- 单客户端串行处理，多客户端会排队。
- `wire.py` 与机器人端的 `examples/g2_pi05_remote/wire.py` 是同一份文件，改动要同步。
