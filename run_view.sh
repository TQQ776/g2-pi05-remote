#!/usr/bin/env bash
# 在开发机上用 x86_64 GDK 直接看机器人相机（混合部署）
# 需要 X11：确保 DISPLAY 已设置（本机实测 DISPLAY=:1）
set -euo pipefail
cd "$(dirname "$0")"

GDK_ROOT="${G2_GDK_ROOT:-/mnt/models/gdk_x86/app}"
if [[ -f "$GDK_ROOT/env.sh" ]]; then
    # env.sh 设置 LD_LIBRARY_PATH / PYTHONPATH / APP_CONF_PATH / LOCATOR_IP / AORTA_DISCOVERY_URI。
    # LD_LIBRARY_PATH 必须由 shell 在进程启动前设置，进程内再改是无效的。
    # env.sh 把第一个参数当成安装目录（并且引用了 $1，我们开着 set -u），显式传进去
    set +u
    # shellcheck disable=SC1090
    source "$GDK_ROOT/env.sh" "$GDK_ROOT" >/dev/null
    set -u
else
    echo "找不到 $GDK_ROOT/env.sh，请先解压 server_installer.tar.gz 或设置 G2_GDK_ROOT" >&2
    exit 1
fi

# GDK 扩展是 CPython 3.10 的 ABI，不能用系统默认的 3.12
exec /usr/bin/python3.10 view.py "$@"
