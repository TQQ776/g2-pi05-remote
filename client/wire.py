"""极简 TCP 消息协议：4 字节大端长度 + JSON 头 + 原始字节块。

只依赖标准库和 numpy，机器人上的 python 3.10 / 服务器上的 python 3.11 都能直接跑，
不需要在机器人侧安装 msgpack / websockets 之类的额外依赖。

这不是一个通用序列化库：只支持「一层字典 + numpy 数组 + 不透明字节块」，
够传 state / 图像 / action 用，也刻意不支持嵌套结构和 pickle（避免反序列化风险）。

服务器端 openpi/server/wire.py 是同一个文件，改动需要同步。
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np

_LEN = struct.Struct(">I")
MAX_HEADER = 1 << 20          # 1 MiB，头部只有 JSON，不该这么大
MAX_TOTAL = 128 << 20         # 128 MiB，三路 1080p JPEG 远小于这个


def pack(meta: dict, arrays: dict | None = None, blobs: dict | None = None) -> bytes:
    """把一条消息编码成可发送的字节串。

    arrays 里的值必须是 numpy 数组，按 dtype/shape 原样传输；
    blobs 里的值当作不透明字节（JPEG 原文、JSON 片段等）。
    """
    spec_arrays: list[dict] = []
    spec_blobs: list[dict] = []
    payload: list[bytes] = []

    for key, value in (arrays or {}).items():
        arr = np.ascontiguousarray(value)
        raw = arr.tobytes()
        spec_arrays.append(
            {"key": key, "dtype": arr.dtype.str, "shape": list(arr.shape), "n": len(raw)}
        )
        payload.append(raw)

    for key, value in (blobs or {}).items():
        raw = bytes(value)
        spec_blobs.append({"key": key, "n": len(raw)})
        payload.append(raw)

    header = json.dumps({"meta": meta, "arrays": spec_arrays, "blobs": spec_blobs}).encode("utf-8")
    if len(header) > MAX_HEADER:
        raise ValueError(f"消息头过大: {len(header)} 字节")
    return _LEN.pack(len(header)) + header + b"".join(payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("对端关闭了连接")
        got += k
    return bytes(buf)


def recv(sock: socket.socket) -> tuple[dict, dict, dict]:
    """读一条完整消息，返回 (meta, arrays, blobs)。"""
    (header_len,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    if header_len > MAX_HEADER:
        raise ValueError(f"消息头过大: {header_len} 字节，协议不匹配？")
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))

    total = sum(spec["n"] for spec in header["arrays"]) + sum(spec["n"] for spec in header["blobs"])
    if total > MAX_TOTAL:
        raise ValueError(f"消息体过大: {total} 字节")

    arrays: dict = {}
    for spec in header["arrays"]:
        raw = _recv_exact(sock, spec["n"])
        arr = np.frombuffer(raw, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])
        arrays[spec["key"]] = arr.copy()

    blobs: dict = {}
    for spec in header["blobs"]:
        blobs[spec["key"]] = _recv_exact(sock, spec["n"])

    return header["meta"], arrays, blobs


def send(sock: socket.socket, meta: dict, arrays: dict | None = None, blobs: dict | None = None) -> None:
    sock.sendall(pack(meta, arrays, blobs))
