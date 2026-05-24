#!/bin/bash

# multi_pip_install.sh - 批量精准安装Python包 (极简版)
# 用法：./multi_pip_install.sh [python_path]
# 遇到任何错误会立即退出。

set -euo pipefail  # 启用严格模式，任何错误立即退出

# --- 配置区 ---
PYTHON=${1:-python3}
TIMEOUT=300

# 关键包列表
KEY_PACKAGES=(
    "transformers==4.49.0"  # NOTE transformers==4.53.1在load language模型参数时候会有问题
    "diffusers==0.29.1"
    "torch==2.5.1+cu124"
    "torchvision==0.20.1+cu124"
    "torchaudio==2.5.1+cu124"
    "gradio==5.35"
)

# --- 主流程 ---
# 卸载pynvml（如果存在）
echo ">>> 开始卸载pynvml..."
$PYTHON -m pip uninstall -y pynvml || true

# 从requirements.txt安装所有包
echo ">>> 开始从requirements.txt安装软件包..."
timeout $TIMEOUT $PYTHON -m pip install --upgrade --no-cache-dir -r requirements.txt

# 单独安装关键包
echo ">>> 开始安装关键软件包..."
for pkg in "${KEY_PACKAGES[@]}"; do
    echo "--- 正在安装: $pkg ---"
    timeout $TIMEOUT $PYTHON -m pip install --upgrade --no-cache-dir "$pkg"
done

# 3. 成功结束
echo "✓ 所有包均已成功安装或更新。"
