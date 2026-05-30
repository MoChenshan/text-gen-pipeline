#!/bin/bash
# ============================================================
# 推理服务部署脚本（支持 vLLM、SGLang 和 Transformers）
# 用途: 在 AutoDL 上部署合并后的模型，提供 OpenAI 兼容 API
# 使用: bash deploy_vllm.sh [sglang|vllm|transformers]
# ============================================================

set -e

# ---- 默认配置 ----
ENGINE=${1:-"transformers"}  # 推理引擎: sglang, vllm 或 transformers
PORT=6006
MODEL_PATH="/root/autodl-tmp/outputs/qwen3.6-27b-merged"
MODEL_NAME="qwen3.6-27b-nsfw"
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90
LOG_FILE="/root/autodl-tmp/outputs/inference_server.log"

echo "=========================================="
echo "  推理服务部署"
echo "=========================================="
echo "  推理引擎: ${ENGINE}"
echo "  模型路径: ${MODEL_PATH}"
echo "  模型名称: ${MODEL_NAME}"
echo "  端口: ${PORT}"
echo "  最大序列长度: ${MAX_MODEL_LEN}"
echo "  GPU 显存利用率: ${GPU_MEMORY_UTILIZATION}"
echo ""

# ---- 检查模型 ----
if [ ! -d "${MODEL_PATH}" ]; then
    echo "错误: 模型路径不存在: ${MODEL_PATH}"
    echo "请先运行 merge_lora.py 合并模型"
    exit 1
fi

# ---- 停止已有服务 ----
tmux kill-session -t inference 2>/dev/null || true
sleep 1

# ---- 显示 GPU 信息 ----
echo "[GPU 信息]"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ---- 根据引擎启动服务 ----
if [ "${ENGINE}" = "sglang" ]; then
    echo "[启动] SGLang OpenAI 兼容 API 服务..."
    echo "  API 地址: http://0.0.0.0:${PORT}/v1"
    echo ""

    # SGLang 启动命令（参考 Qwen3.6 官方 README）
    # --reasoning-parser qwen3: 支持 thinking 模式解析
    # --mem-fraction-static: GPU 显存静态分配比例
    tmux new-session -d -s inference "python -m sglang.launch_server \
        --model-path ${MODEL_PATH} \
        --served-model-name ${MODEL_NAME} \
        --port ${PORT} \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --mem-fraction-static ${GPU_MEMORY_UTILIZATION} \
        --context-length ${MAX_MODEL_LEN} \
        --reasoning-parser qwen3 \
        --trust-remote-code \
    2>&1 | tee ${LOG_FILE}"

elif [ "${ENGINE}" = "vllm" ]; then
    echo "[启动] vLLM OpenAI 兼容 API 服务..."
    echo "  API 地址: http://0.0.0.0:${PORT}/v1"
    echo ""

    # vLLM 启动命令（参考 Qwen3.6 官方 README）
    # --language-model-only: 跳过视觉编码器，节省显存（纯文本推理）
    # --reasoning-parser qwen3: 支持 thinking 模式解析
    tmux new-session -d -s inference "vllm serve ${MODEL_PATH} \
        --served-model-name ${MODEL_NAME} \
        --port ${PORT} \
        --host 0.0.0.0 \
        --dtype bfloat16 \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --reasoning-parser qwen3 \
        --language-model-only \
        --trust-remote-code \
    2>&1 | tee ${LOG_FILE}"

elif [ "${ENGINE}" = "transformers" ]; then
    echo "[启动] Transformers 原生推理 API 服务..."
    echo "  API 地址: http://0.0.0.0:${PORT}/v1"
    echo "  注意: 吞吐量低于 vLLM/SGLang，但兼容性最好"
    echo ""

    # 获取脚本所在目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    tmux new-session -d -s inference "python ${SCRIPT_DIR}/serve_transformers.py \
        --model-path ${MODEL_PATH} \
        --model-name ${MODEL_NAME} \
        --port ${PORT} \
        --host 0.0.0.0 \
    2>&1 | tee ${LOG_FILE}"

else
    echo "错误: 未知引擎 '${ENGINE}'"
    echo "支持的引擎: sglang, vllm, transformers"
    exit 1
fi

echo "=========================================="
echo "  推理服务已在后台启动!"
echo "=========================================="
echo ""
echo "常用命令:"
echo "  查看服务: tmux attach -t inference"
echo "  查看日志: tail -f ${LOG_FILE}"
echo "  停止服务: tmux kill-session -t inference"
echo ""
echo "测试命令:"
echo "  curl http://localhost:${PORT}/v1/models"
echo ""
echo "本地 SSH 隧道（在本地终端执行）:"
echo "  ssh -p 42655 -L ${PORT}:localhost:${PORT} root@connect.westd.seetacloud.com -N"
echo ""
echo "然后本地访问: http://localhost:${PORT}/v1"
echo ""
