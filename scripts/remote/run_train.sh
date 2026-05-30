#!/bin/bash
# ============================================================
# 训练启动脚本（自动适配单卡/多卡）
# 用途: 启动 LLaMA-Factory LoRA 微调训练
#
# 使用方式:
#   bash run_train.sh           # 自动检测 GPU 数量
#   bash run_train.sh 1         # 强制单卡训练
#   bash run_train.sh 2         # 强制双卡训练
#   bash run_train.sh 4         # 强制4卡训练
#   bash run_train.sh 6         # 强制6卡训练
#   bash run_train.sh 4 0,1,2,3 # 指定使用 GPU
# ============================================================

set -e

# ---- 参数解析 ----
NUM_GPUS=${1:-"auto"}
GPU_IDS=${2:-""}

# ---- 路径配置 ----
PROJECT_DIR="/root/project"
LLAMA_FACTORY_DIR="${PROJECT_DIR}/LLaMA-Factory"
CONFIG_DIR="${PROJECT_DIR}/text-gen-pipeline/configs"
OUTPUT_DIR="/root/autodl-tmp/outputs"
LOG_FILE="${OUTPUT_DIR}/train.log"

# ---- 自动检测 GPU 数量 ----
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "${NUM_GPUS}" = "auto" ]; then
    NUM_GPUS=${AVAILABLE_GPUS}
    echo "[自动检测] 发现 ${NUM_GPUS} 张 GPU"
fi

# 验证 GPU 数量
if [ ${NUM_GPUS} -gt ${AVAILABLE_GPUS} ]; then
    echo "错误: 请求 ${NUM_GPUS} 张 GPU，但只有 ${AVAILABLE_GPUS} 张可用"
    exit 1
fi

# ---- 选择训练配置 ----
case ${NUM_GPUS} in
    1)
        TRAIN_CONFIG="${CONFIG_DIR}/qwen3.6_27b_lora_sft.yaml"
        TRAIN_MODE="单卡训练"
        ;;
    2)
        TRAIN_CONFIG="${CONFIG_DIR}/qwen3.6_27b_lora_sft_2gpu.yaml"
        TRAIN_MODE="双卡训练 (DeepSpeed ZeRO-2)"
        ;;
    4)
        TRAIN_CONFIG="${CONFIG_DIR}/qwen3.6_27b_lora_sft_4gpu.yaml"
        TRAIN_MODE="4卡训练 (DeepSpeed ZeRO-2)"
        ;;
    6)
        TRAIN_CONFIG="${CONFIG_DIR}/qwen3.6_27b_lora_sft_6gpu.yaml"
        TRAIN_MODE="6卡训练 (DeepSpeed ZeRO-2)"
        ;;
    *)
        TRAIN_CONFIG="${CONFIG_DIR}/qwen3.6_27b_lora_sft_2gpu.yaml"
        TRAIN_MODE="多卡训练 (${NUM_GPUS} GPUs, DeepSpeed ZeRO-2)"
        echo "警告: 无专用 ${NUM_GPUS} 卡配置，使用2卡配置（需手动调整 gradient_accumulation_steps）"
        ;;
esac

echo "=========================================="
echo "  启动 Qwen3.6-27B LoRA 训练"
echo "  模式: ${TRAIN_MODE}"
echo "=========================================="

# ---- 检查环境 ----
echo ""
echo "[检查] 验证环境..."

# 检查模型是否存在
if [ ! -d "/root/autodl-tmp/models/Qwen3.6-27B" ]; then
    echo "错误: 模型不存在，请先运行 setup_env.sh"
    exit 1
fi

# 检查训练数据是否存在
if [ ! -f "/root/autodl-tmp/datasets/final/train_mixed.jsonl" ]; then
    echo "错误: 训练数据不存在，请先运行数据准备流程"
    echo "  1. python process_proprietary_datasets.py"
    echo "  2. bash download_general_datasets.sh"
    echo "  3. python convert_general_datasets.py"
    echo "  4. python mix_datasets.py --total_samples 300000"
    exit 1
fi

# 检查 LLaMA-Factory 是否安装
if ! command -v llamafactory-cli &> /dev/null; then
    echo "错误: llamafactory-cli 未找到，请先运行 setup_env.sh"
    exit 1
fi

echo "  ✓ 模型存在"
echo "  ✓ 训练数据存在"
echo "  ✓ LLaMA-Factory 已安装"

# ---- 注册数据集到 LLaMA-Factory ----
echo ""
echo "[准备] 注册数据集..."
DATASET_INFO_SRC="${CONFIG_DIR}/dataset_info.json"
DATASET_INFO_DST="${LLAMA_FACTORY_DIR}/data/dataset_info.json"

python << 'EOF'
import json

src_path = "/root/project/text-gen-pipeline/configs/dataset_info.json"
dst_path = "/root/project/LLaMA-Factory/data/dataset_info.json"

# 读取 LLaMA-Factory 原有的 dataset_info
with open(dst_path, "r", encoding="utf-8") as f:
    original = json.load(f)

# 读取我们的自定义数据集信息
with open(src_path, "r", encoding="utf-8") as f:
    custom = json.load(f)

# 合并
original.update(custom)

# 写回
with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(original, f, ensure_ascii=False, indent=2)

print("  ✓ 数据集注册完成")
EOF

# ---- 创建输出目录 ----
mkdir -p ${OUTPUT_DIR}

# ---- 显示训练配置 ----
echo ""
echo "[配置] 训练参数:"
echo "  配置文件: ${TRAIN_CONFIG}"
echo "  训练模式: ${TRAIN_MODE}"
echo "  模型: Qwen3.6-27B (BF16)"
echo "  方法: LoRA (rank=64, alpha=128, target=all)"
echo "  序列长度: 4096"
case ${NUM_GPUS} in
    1) echo "  有效Batch Size: 2 × 8 (grad_accum) = 16" ;;
    2) echo "  有效Batch Size: 1 × 16 (grad_accum) × 2 (GPUs) = 32" ;;
    4) echo "  有效Batch Size: 1 × 8 (grad_accum) × 4 (GPUs) = 32" ;;
    6) echo "  有效Batch Size: 1 × 6 (grad_accum) × 6 (GPUs) = 36" ;;
    *) echo "  有效Batch Size: 见配置文件" ;;
esac
echo "  学习率: 5e-4 (cosine scheduler)"
echo "  Epochs: 1"
echo "  Gradient Checkpointing: 开启"
echo ""

# ---- 检查 GPU ----
echo "[GPU] 显卡信息:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# ---- 构建训练命令 ----
if [ ${NUM_GPUS} -eq 1 ]; then
    # === 单卡训练 ===
    if [ -n "${GPU_IDS}" ]; then
        TRAIN_CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} llamafactory-cli train ${TRAIN_CONFIG}"
    else
        TRAIN_CMD="llamafactory-cli train ${TRAIN_CONFIG}"
    fi
else
    # === 多卡训练 (DeepSpeed) ===
    if [ -n "${GPU_IDS}" ]; then
        export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
        echo "  使用 GPU: ${GPU_IDS}"
    fi
    
    # 使用 deepspeed launcher 或 torchrun
    # LLaMA-Factory 内部会处理 DeepSpeed，只需设置好环境变量
    TRAIN_CMD="FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=${NUM_GPUS} llamafactory-cli train ${TRAIN_CONFIG}"
fi

echo "[命令] ${TRAIN_CMD}"
echo ""

# ---- 启动训练 ----
echo "[训练] 开始训练..."
echo "  使用 tmux 运行，可以安全断开 SSH"
echo "  会话名称: train"
echo ""

# 杀掉已有的 train 会话（如果存在）
tmux kill-session -t train 2>/dev/null || true

# 使用 tmux 启动训练（防止 SSH 断开导致训练中断）
tmux new-session -d -s train "cd ${LLAMA_FACTORY_DIR} && ${TRAIN_CMD} 2>&1 | tee ${LOG_FILE}; echo ''; echo '训练结束! 按任意键退出...'; read"

# 等待一下确认启动成功
sleep 3

# 检查 tmux 会话是否存在
if tmux has-session -t train 2>/dev/null; then
    echo "=========================================="
    echo "  ✓ 训练已在后台启动!"
    echo "=========================================="
else
    echo "=========================================="
    echo "  ✗ 训练启动失败! 请检查日志"
    echo "=========================================="
    cat ${LOG_FILE} 2>/dev/null | tail -20
    exit 1
fi

echo ""
echo "常用命令:"
echo "  查看训练: tmux attach -t train"
echo "  查看日志: tail -f ${LOG_FILE}"
echo "  查看GPU:  watch -n 1 nvidia-smi"
echo "  停止训练: tmux kill-session -t train"
echo ""
echo "训练完成后:"
echo "  1. 合并 LoRA: python merge_lora.py"
echo "  2. 部署推理: bash deploy_vllm.sh"
echo ""
