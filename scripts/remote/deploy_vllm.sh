#!/bin/bash
# ============================================================
# vLLM 推理服务部署脚本
# 用途: 在 AutoDL 上部署合并后的模型，提供 OpenAI 兼容 API
# 使用: bash deploy_vllm.sh [--port 6006] [--model_path PATH]
# ============================================================

set -e

# ---- 默认配置 ----
PORT=${1:-6006}
MODEL_PATH=${2:-"/root/autodl-tmp/outputs/qwen3.6-27b-merged"}
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90

echo "=========================================="
echo "  vLLM 推理服务部署"
echo "=========================================="
echo "  模型路径: ${MODEL_PATH}"
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

# ---- 检查 vLLM ----
if ! python -c "import vllm" 2>/dev/null; then
    echo "安装 vLLM..."
    pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

# ---- 显示 GPU 信息 ----
echo "[GPU 信息]"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ---- 启动 vLLM 服务 ----
echo "[启动] vLLM OpenAI 兼容 API 服务..."
echo "  API 地址: http://0.0.0.0:${PORT}/v1"
echo "  模型名称: qwen3.6-27b-nsfw"
echo ""
echo "  测试命令:"
echo "    curl http://localhost:${PORT}/v1/models"
echo ""
echo "  使用 tmux 运行，可安全断开 SSH"
echo "  查看服务: tmux attach -t vllm"
echo ""

# 使用 tmux 启动（防止 SSH 断开）
tmux new-session -d -s vllm "python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name qwen3.6-27b-nsfw \
    --dtype bfloat16 \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --trust-remote-code \
2>&1 | tee /root/autodl-tmp/outputs/vllm_server.log"

echo "=========================================="
echo "  vLLM 服务已在后台启动!"
echo "=========================================="
echo ""
echo "常用命令:"
echo "  查看服务: tmux attach -t vllm"
echo "  查看日志: tail -f /root/autodl-tmp/outputs/vllm_server.log"
echo "  停止服务: tmux kill-session -t vllm"
echo ""
echo "AutoDL 端口映射:"
echo "  在 AutoDL 控制台 -> 自定义服务 -> 开放端口 ${PORT}"
echo "  获取外部访问地址后，配置到 SillyTavern 中"
echo ""
