"""
Text Generation Pipeline - 主入口
==================================

基于 Qwen3.6-27B + LoRA 的文本生成微调 Pipeline

项目结构:
    text-gen-pipeline/
    ├── main.py                          # 本文件 - Pipeline 总览与快速命令
    ├── configs/                         # 训练配置
    │   ├── qwen3.6_27b_lora_sft.yaml   # 单卡训练配置
    │   ├── qwen3.6_27b_lora_sft_2gpu.yaml  # 双卡训练配置
    │   ├── ds_z2_config.json            # DeepSpeed ZeRO-2 配置
    │   └── dataset_info.json            # LLaMA-Factory 数据集注册
    ├── scripts/
    │   ├── remote/                      # 远端(AutoDL)执行的脚本
    │   │   ├── setup_env.sh             # 环境初始化
    │   │   ├── process_proprietary_datasets.py  # 专有数据集下载+预处理
    │   │   ├── download_general_datasets.sh  # 通用数据集下载
    │   │   ├── convert_general_datasets.py   # 通用数据格式转换
    │   │   ├── mix_datasets.py          # 数据集混合
    │   │   ├── run_train.sh             # 启动训练
    │   │   ├── merge_lora.py            # LoRA 合并
    │   │   └── deploy_vllm.sh           # vLLM 部署
    │   └── local/                       # 本地(Windows)执行的脚本
    │       ├── sync_to_remote.bat       # 同步到远端
    │       ├── download_lora.bat        # 下载 LoRA 权重
    │       └── test_api.py              # API 测试
    ├── data/
    │   ├── raw/                         # 原始专有数据（HF自动下载）
    │   └── processed/                   # 处理后的专有数据
    └── outputs/                         # 本地输出（LoRA权重备份等）

完整流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  Phase 1: 环境准备                                          │
    │    [远端] bash setup_env.sh                                 │
    │    [远端] bash download_general_datasets.sh                 │
    │                                                             │
    │  Phase 2: 数据处理                                          │
    │    [远端] python process_proprietary_datasets.py            │
    │    [远端] python convert_general_datasets.py                │
    │    [远端] python mix_datasets.py                            │
    │                                                             │
    │  Phase 3: 训练                                              │
    │    [远端] bash run_train.sh                                 │
    │    [远端] 监控训练: tmux attach -t train                    │
    │                                                             │
    │  Phase 4: 部署                                              │
    │    [远端] python merge_lora.py                              │
    │    [远端] bash deploy_vllm.sh                               │
    │    [本地] python test_api.py --api_url http://...           │
    │                                                             │
    │  Phase 5: 使用                                              │
    │    [本地] SillyTavern 连接远端 API                           │
    │    [手机] 浏览器访问 SillyTavern / Open WebUI               │
    └─────────────────────────────────────────────────────────────┘
"""

import os
import sys


def print_pipeline_status():
    """打印 Pipeline 当前状态"""
    print("=" * 60)
    print("  Text Generation Pipeline - Qwen3.6-27B LoRA")
    print("=" * 60)
    print()
    
    # 检查本地文件状态
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    checks = [
        ("configs/qwen3.6_27b_lora_sft.yaml", "训练配置"),
        ("configs/dataset_info.json", "数据集注册"),
        ("scripts/remote/setup_env.sh", "环境初始化脚本"),
        ("scripts/remote/process_proprietary_datasets.py", "专有数据预处理脚本"),
        ("scripts/remote/download_general_datasets.sh", "通用数据集下载脚本"),
        ("scripts/remote/convert_general_datasets.py", "通用数据格式转换脚本"),
        ("scripts/remote/mix_datasets.py", "数据集混合脚本"),
        ("scripts/remote/run_train.sh", "训练启动脚本"),
        ("scripts/remote/merge_lora.py", "LoRA 合并脚本"),
        ("scripts/remote/deploy_vllm.sh", "vLLM 部署脚本"),
        ("scripts/local/sync_to_remote.bat", "同步脚本"),
        ("scripts/local/test_api.py", "API 测试脚本"),
    ]
    
    print("  [文件状态检查]")
    print()
    for path, desc in checks:
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            if os.path.isdir(full_path):
                files = os.listdir(full_path)
                if files:
                    print(f"  ✓ {desc}: {path} ({len(files)} 个文件)")
                else:
                    print(f"  ○ {desc}: {path} (空目录，待准备)")
            else:
                print(f"  ✓ {desc}: {path}")
        else:
            print(f"  ✗ {desc}: {path} (不存在)")
    
    print()
    print("  [下一步操作]")
    print()
    print("  1. 修改 scripts/local/sync_to_remote.bat 中的 SSH 配置")
    print("  2. 在 AutoDL 上运行 setup_env.sh 初始化环境")
    print("  3. 运行 process_proprietary_datasets.py 下载并处理专有数据")
    print("  4. 运行 download_general_datasets.sh 下载通用数据")
    print("  5. 运行 mix_datasets.py 混合数据集")
    print("  6. 运行 run_train.sh 启动训练")
    print()
    print("=" * 60)


if __name__ == "__main__":
    print_pipeline_status()