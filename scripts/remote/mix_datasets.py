"""
数据集混合脚本
将专有数据集和通用数据集按指定比例混合，生成最终训练数据

混合比例（默认）:
  - 专有 NSFW Novel: 65%
  - 中文通用对话/指令: 25%
  - 中文创意写作: 10%（如有）

使用方法:
    python mix_datasets.py [--proprietary_ratio 0.65] [--general_ratio 0.25] [--creative_ratio 0.10]
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

# ---- 路径配置 ----
DATASET_DIR = "/root/autodl-fs/datasets"
PROPRIETARY_DIR = os.path.join(DATASET_DIR, "proprietary", "converted")
GENERAL_DIR = os.path.join(DATASET_DIR, "general", "converted")
OUTPUT_DIR = os.path.join(DATASET_DIR, "final")


def load_json_dataset(filepath: str) -> List[Dict]:
    """加载 JSON 格式数据集"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_dataset(data: List[Dict], target_count: int) -> List[Dict]:
    """
    从数据集中采样指定数量
    如果目标数量大于数据量，则重复采样（过采样）
    如果目标数量小于数据量，则随机采样（欠采样）
    """
    if target_count <= 0:
        return []
    
    if target_count <= len(data):
        return random.sample(data, target_count)
    else:
        # 过采样：重复数据
        result = data.copy()
        while len(result) < target_count:
            result.extend(random.sample(data, min(len(data), target_count - len(result))))
        return result[:target_count]


def collect_json_files(directory: str) -> List[Dict]:
    """收集目录下所有 JSON 文件的数据"""
    all_data = []
    if not os.path.exists(directory):
        print(f"  警告: 目录不存在 {directory}")
        return all_data
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            data = load_json_dataset(filepath)
            print(f"    - {filename}: {len(data)} 条")
            all_data.extend(data)
    
    return all_data


def main():
    parser = argparse.ArgumentParser(description="数据集混合脚本")
    parser.add_argument("--proprietary_ratio", type=float, default=0.65,
                        help="专有数据集比例 (默认: 0.65)")
    parser.add_argument("--general_ratio", type=float, default=0.25,
                        help="通用数据集比例 (默认: 0.25)")
    parser.add_argument("--creative_ratio", type=float, default=0.10,
                        help="创意写作数据集比例 (默认: 0.10)")
    parser.add_argument("--total_samples", type=int, default=None,
                        help="最终数据集总条数 (默认: 以专有数据集为基准自动计算)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 验证比例
    total_ratio = args.proprietary_ratio + args.general_ratio + args.creative_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"警告: 比例之和为 {total_ratio}，不等于 1.0，将自动归一化")
        args.proprietary_ratio /= total_ratio
        args.general_ratio /= total_ratio
        args.creative_ratio /= total_ratio
    
    print("========================================")
    print("  数据集混合")
    print("========================================")
    print(f"  专有数据比例: {args.proprietary_ratio:.0%}")
    print(f"  通用数据比例: {args.general_ratio:.0%}")
    print(f"  创意写作比例: {args.creative_ratio:.0%}")
    print("")
    
    # ---- 加载专有数据集 ----
    print("[1/3] 加载专有数据集...")
    proprietary_data = collect_json_files(PROPRIETARY_DIR)
    print(f"  总计: {len(proprietary_data)} 条\n")
    
    if len(proprietary_data) == 0:
        print("错误: 专有数据集为空！请先准备专有数据集。")
        print(f"  期望路径: {PROPRIETARY_DIR}/*.json")
        return
    
    # ---- 加载通用数据集 ----
    print("[2/3] 加载通用数据集...")
    general_data = collect_json_files(GENERAL_DIR)
    print(f"  总计: {len(general_data)} 条\n")
    
    # ---- 加载创意写作数据集（可选） ----
    print("[3/3] 加载创意写作数据集...")
    creative_dir = os.path.join(DATASET_DIR, "creative", "converted")
    creative_data = collect_json_files(creative_dir)
    print(f"  总计: {len(creative_data)} 条\n")
    
    # ---- 计算采样数量 ----
    if args.total_samples:
        total = args.total_samples
    else:
        # 以专有数据集为基准，计算总量
        total = int(len(proprietary_data) / args.proprietary_ratio)
    
    proprietary_count = int(total * args.proprietary_ratio)
    general_count = int(total * args.general_ratio)
    creative_count = int(total * args.creative_ratio)
    
    # 如果没有创意写作数据，将其比例分配给通用数据
    if len(creative_data) == 0 and creative_count > 0:
        print("  注意: 无创意写作数据，将比例分配给通用数据")
        general_count += creative_count
        creative_count = 0
    
    print(f"采样计划:")
    print(f"  专有数据: {proprietary_count} 条 (源: {len(proprietary_data)})")
    print(f"  通用数据: {general_count} 条 (源: {len(general_data)})")
    print(f"  创意写作: {creative_count} 条 (源: {len(creative_data)})")
    print(f"  总计: {proprietary_count + general_count + creative_count} 条")
    print("")
    
    # ---- 采样 ----
    final_data = []
    final_data.extend(sample_dataset(proprietary_data, proprietary_count))
    final_data.extend(sample_dataset(general_data, general_count))
    if creative_count > 0:
        final_data.extend(sample_dataset(creative_data, creative_count))
    
    # ---- 打乱 ----
    random.shuffle(final_data)
    
    # ---- 保存 ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "train_mixed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"========================================")
    print(f"  混合完成!")
    print(f"  最终数据集: {len(final_data)} 条")
    print(f"  输出路径: {output_path}")
    print(f"========================================")
    
    # ---- 生成数据集信息文件 ----
    info = {
        "total_samples": len(final_data),
        "proprietary_count": proprietary_count,
        "general_count": general_count,
        "creative_count": creative_count,
        "proprietary_ratio": args.proprietary_ratio,
        "general_ratio": args.general_ratio,
        "creative_ratio": args.creative_ratio,
        "seed": args.seed,
    }
    info_path = os.path.join(OUTPUT_DIR, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  数据集信息: {info_path}")


if __name__ == "__main__":
    main()
