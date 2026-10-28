#!/usr/bin/env bash
# 启动 π0.5 远程推理控制
#
# 两种部署共用这一个脚本，自动判断：
#   1. 机器人本体：系统 python 就是 3.10 + agibot_gdk，直接跑
#   2. 混合部署（开发机）：有 G2_GDK_ROOT/env.sh 和 /usr/bin/python3.10 时自动切过去
set -euo pipefail
cd "$(dirname "$0")"

GDK_ROOT="${G2_GDK_ROOT:-/mnt/models/gdk_x86/app}"

if [[ -f "$GDK_ROOT/env.sh" && -x /usr/bin/python3.10 ]]; then
    # env.sh 里引用了 $1，我们开着 set -u，得先关掉再 source
    # env.sh 把第一个参数当成自己的安装目录，必须显式传，否则会吃掉我们的 --help
    set +u
    # shellcheck disable=SC1090
    source "$GDK_ROOT/env.sh" "$GDK_ROOT" >/dev/null
    set -u
    # LD_LIBRARY_PATH 必须由 shell 在进程启动前设置，进程内再改是无效的
    exec /usr/bin/python3.10 client.py "$@"
fi

exec python3 client.py "$@"
