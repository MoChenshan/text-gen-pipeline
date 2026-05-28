#!/bin/bash
# ============================================================
# 通用数据集下载脚本（远端执行）
# 用途: 从 HuggingFace/ModelScope 拉取中文通用数据集
# 使用: bash download_general_datasets.sh
# ============================================================

set -e

echo "=========================================="
echo "  下载通用数据集"
echo "=========================================="

DATASET_DIR="/root/autodl-fs/datasets/general"
mkdir -p ${DATASET_DIR}

# ---- 使用 modelscope 下载（国内速度快） ----
pip install modelscope datasets -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null

python << 'EOF'
import os
import json
from pathlib import Path

DATASET_DIR = "/root/autodl-fs/datasets/general"

def download_from_modelscope():
    """从 ModelScope 下载中文通用数据集"""
    from modelscope.msdatasets import MsDataset
    
    datasets_to_download = {
        # ---- 中文通用对话/指令数据 ----
        "firefly": {
            "name": "YeungNLP/firefly-train-1.1M",
            "subset": None,
            "split": "train",
            "description": "中文多任务指令数据，覆盖23种NLP任务",
            "sample_ratio": 0.05,  # 取5%，约5.5万条
        },
        # ---- 中文高质量指令数据 ----
        "coig_cqia": {
            "name": "m-a-p/COIG-CQIA",
            "subset": None,
            "split": "train",
            "description": "高质量中文指令数据（知乎、豆瓣等来源）",
            "sample_ratio": 0.1,  # 取10%
        },
    }
    
    return datasets_to_download

def download_from_huggingface():
    """从 HuggingFace 下载数据集（备用方案）"""
    from datasets import load_dataset
    
    print("\n[1/3] 下载 firefly 中文指令数据...")
    firefly_path = os.path.join(DATASET_DIR, "firefly")
    if not os.path.exists(firefly_path):
        os.makedirs(firefly_path, exist_ok=True)
        try:
            ds = load_dataset("YeungNLP/firefly-train-1.1M", split="train", streaming=True)
            # 采样约5万条
            samples = []
            for i, item in enumerate(ds):
                if i >= 50000:
                    break
                samples.append(item)
            
            # 保存为 JSON
            with open(os.path.join(firefly_path, "firefly_50k.json"), "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            print(f"  ✓ firefly 下载完成: {len(samples)} 条")
        except Exception as e:
            print(f"  ✗ firefly 下载失败: {e}")
    else:
        print(f"  - firefly 已存在，跳过")

    print("\n[2/3] 下载 COIG-CQIA 中文指令数据...")
    coig_path = os.path.join(DATASET_DIR, "coig_cqia")
    if not os.path.exists(coig_path):
        os.makedirs(coig_path, exist_ok=True)
        try:
            # COIG-CQIA 有多个子集，选择几个高质量的
            subsets = ["zhihu", "lk", "human_value"]
            all_samples = []
            for subset in subsets:
                try:
                    ds = load_dataset("m-a-p/COIG-CQIA", subset, split="train", streaming=True)
                    count = 0
                    for item in ds:
                        if count >= 10000:  # 每个子集取1万条
                            break
                        all_samples.append(item)
                        count += 1
                    print(f"    - {subset}: {count} 条")
                except Exception as e:
                    print(f"    - {subset} 失败: {e}")
            
            with open(os.path.join(coig_path, "coig_cqia_30k.json"), "w", encoding="utf-8") as f:
                json.dump(all_samples, f, ensure_ascii=False, indent=2)
            print(f"  ✓ COIG-CQIA 下载完成: {len(all_samples)} 条")
        except Exception as e:
            print(f"  ✗ COIG-CQIA 下载失败: {e}")
    else:
        print(f"  - COIG-CQIA 已存在，跳过")

    print("\n[3/3] 下载 Alpaca-GPT4-zh 中文指令数据...")
    alpaca_path = os.path.join(DATASET_DIR, "alpaca_zh")
    if not os.path.exists(alpaca_path):
        os.makedirs(alpaca_path, exist_ok=True)
        try:
            ds = load_dataset("shibing624/alpaca-zh", split="train")
            samples = [item for item in ds]
            with open(os.path.join(alpaca_path, "alpaca_zh.json"), "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Alpaca-zh 下载完成: {len(samples)} 条")
        except Exception as e:
            print(f"  ✗ Alpaca-zh 下载失败: {e}")
    else:
        print(f"  - Alpaca-zh 已存在，跳过")


if __name__ == "__main__":
    print("使用 HuggingFace datasets 下载...")
    print("(如果速度慢，请设置 HF_ENDPOINT=https://hf-mirror.com)")
    print("")
    
    # 设置镜像（国内加速）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    download_from_huggingface()
    
    print("\n========================================")
    print("  通用数据集下载完成!")
    print(f"  存储路径: {DATASET_DIR}")
    print("========================================")
    print("\n目录内容:")
    for item in os.listdir(DATASET_DIR):
        item_path = os.path.join(DATASET_DIR, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            print(f"  {item}/: {files}")

EOF

echo ""
echo "通用数据集下载脚本执行完毕"
echo "下一步: 运行数据格式转换脚本将数据转为 LLaMA-Factory 格式"
