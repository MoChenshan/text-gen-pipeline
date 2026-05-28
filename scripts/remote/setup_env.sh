#!/bin/bash
# ============================================================
# AutoDL 远端环境初始化脚本
# 用途: 初始化训练环境，安装 LLaMA-Factory，下载基座模型
# 使用: bash setup_env.sh
# ============================================================

set -e

echo "=========================================="
echo "  AutoDL 环境初始化 - Text Gen Pipeline"
echo "=========================================="

# ---- 基础路径配置 ----
WORK_DIR="/root/autodl-tmp"
PERSIST_DIR="/root/autodl-fs"
MODEL_DIR="${WORK_DIR}/models"
OUTPUT_DIR="${PERSIST_DIR}/outputs"
DATASET_DIR="${PERSIST_DIR}/datasets"
PROJECT_DIR="/root/project"

# 创建目录结构
mkdir -p ${MODEL_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${DATASET_DIR}
mkdir -p ${DATASET_DIR}/proprietary
mkdir -p ${DATASET_DIR}/general
mkdir -p ${PROJECT_DIR}

echo "[1/5] 更新系统并安装基础依赖..."
apt-get update && apt-get install -y git git-lfs tmux htop
git lfs install

echo "[2/5] 创建 conda 环境..."
# AutoDL 通常预装了 conda，如果没有则需要先安装
if command -v conda &> /dev/null; then
    conda create -n llama_factory python=3.11 -y || true
    source activate llama_factory || conda activate llama_factory
else
    echo "conda 未找到，使用系统 Python"
fi

echo "[3/5] 安装 LLaMA-Factory..."
cd ${PROJECT_DIR}
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e ".[torch,metrics]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装额外依赖
pip install deepspeed bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[4/5] 下载 Qwen3.6-27B 基座模型..."
cd ${MODEL_DIR}
if [ ! -d "Qwen3.6-27B" ]; then
    # 使用 modelscope 下载（国内速度更快）
    pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -c "
from modelscope import snapshot_download
snapshot_download(
    'Qwen/Qwen3.6-27B',
    local_dir='${MODEL_DIR}/Qwen3.6-27B',
    revision='master'
)
print('模型下载完成!')
"
else
    echo "模型已存在，跳过下载"
fi

echo "[5/5] 验证环境..."
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  显存: {torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "=========================================="
echo "  环境初始化完成!"
echo "=========================================="
echo ""
echo "目录结构:"
echo "  模型路径: ${MODEL_DIR}/Qwen3.6-27B"
echo "  数据集路径: ${DATASET_DIR}"
echo "  输出路径: ${OUTPUT_DIR}"
echo "  项目路径: ${PROJECT_DIR}"
echo ""
echo "下一步:"
echo "  1. 同步专有数据集到 ${DATASET_DIR}/proprietary/"
echo "  2. 运行 bash download_general_datasets.sh 拉取通用数据集"
echo "  3. 运行训练: bash run_train.sh"
echo ""
