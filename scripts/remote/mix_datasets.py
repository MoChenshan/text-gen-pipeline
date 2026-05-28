"""
数据集混合脚本
将专有数据集和通用数据集按指定比例混合，生成最终训练数据

混合比例（默认）:
  - 专有 NSFW Novel: 65%
  - 中文通用对话/指令: 25%
  - 中文创意写作: 10%（如有）

使用方法:
    python mix_datasets.py --total_samples 300000
    python mix_datasets.py --total_samples 500000 --proprietary_ratio 0.70 --general_ratio 0.30

注意:
    专有数据集文件很大（~118GB），本脚本使用流式随机采样，
    不会一次性加载全部数据到内存。
"""

import os
import json
import random
import argparse
import mmap
from pathlib import Path
from typing import List, Dict, Optional

# ---- 路径配置 ----
# 专有数据集（在 autodl-tmp 上，空间充足）
PROPRIETARY_DIR = "/root/autodl-tmp/datasets/proprietary_converted"
# 通用数据集（在 autodl-tmp 上）
GENERAL_DIR = "/root/autodl-tmp/datasets/general/converted"
# 输出目录（autodl-tmp，空间充足）
OUTPUT_DIR = "/root/autodl-tmp/datasets/final"


def count_json_array_items(filepath: str) -> int:
    """快速统计 JSON 数组中的条目数（不加载全部内容）"""
    print(f"    统计条目数: {filepath}")
    count = 0
    # 通过计算顶层 JSON 对象的数量来估算
    # 每个条目以 {"conversations" 开头
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if '"conversations"' in line and '"from"' not in line:
                count += 1
    return count


def get_json_line_offsets(filepath: str) -> List[int]:
    """
    获取 JSON 数组中每个顶层元素的文件偏移位置
    用于后续随机访问采样
    """
    offsets = []
    depth = 0
    in_string = False
    escape_next = False
    
    with open(filepath, "rb") as f:
        # 使用 mmap 高效扫描大文件
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            # 文件为空或无法 mmap
            return offsets
        
        i = 0
        file_size = mm.size()
        
        while i < file_size:
            byte = mm[i:i+1]
            char = byte.decode("utf-8", errors="ignore")
            
            if escape_next:
                escape_next = False
                i += 1
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                i += 1
                continue
            
            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    if depth == 1:  # 顶层数组中的对象开始
                        offsets.append(i)
                    depth += 1
                elif char == '}':
                    depth -= 1
            
            i += 1
        
        mm.close()
    
    return offsets


def stream_sample_large_json(filepath: str, sample_count: int, seed: int = 42) -> List[Dict]:
    """
    从大型 JSON 数组文件中流式随机采样
    使用 reservoir sampling 算法，内存占用恒定
    """
    print(f"    流式采样 {sample_count} 条 from {os.path.basename(filepath)}...")
    
    rng = random.Random(seed)
    reservoir = []
    count = 0
    
    # 逐条解析 JSON 数组
    with open(filepath, "r", encoding="utf-8") as f:
        # 跳过开头的 [
        content = ""
        depth = 0
        in_string = False
        escape_next = False
        
        for line in f:
            for char in line:
                if escape_next:
                    escape_next = False
                    content += char
                    continue
                
                if char == '\\' and in_string:
                    escape_next = True
                    content += char
                    continue
                
                if char == '"':
                    in_string = not in_string
                    content += char
                    continue
                
                if in_string:
                    content += char
                    continue
                
                # 不在字符串内
                if char == '{':
                    depth += 1
                    content += char
                elif char == '}':
                    depth -= 1
                    content += char
                    if depth == 0 and content.strip():
                        # 完成一个顶层对象
                        try:
                            item = json.loads(content.strip())
                            count += 1
                            
                            # Reservoir sampling
                            if len(reservoir) < sample_count:
                                reservoir.append(item)
                            else:
                                j = rng.randint(0, count - 1)
                                if j < sample_count:
                                    reservoir[j] = item
                            
                            if count % 500000 == 0:
                                print(f"      已扫描 {count} 条...")
                        except json.JSONDecodeError:
                            pass
                        content = ""
                elif char in ' \t\n\r,':
                    if depth > 0:
                        content += char
                    # 顶层的逗号和空白忽略
                elif char == '[' and depth == 0:
                    pass  # 跳过最外层的 [
                elif char == ']' and depth == 0:
                    pass  # 跳过最外层的 ]
                else:
                    content += char
    
    print(f"      扫描完成: 共 {count} 条，采样 {len(reservoir)} 条")
    return reservoir


def load_small_json(filepath: str) -> List[Dict]:
    """加载较小的 JSON 文件（通用数据集，通常 < 1GB）"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_small_json_files(directory: str) -> List[Dict]:
    """收集目录下所有 JSON 文件的数据（适用于小文件）"""
    all_data = []
    if not os.path.exists(directory):
        print(f"  警告: 目录不存在 {directory}")
        return all_data
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            file_size_mb = os.path.getsize(filepath) / 1024 / 1024
            print(f"    - {filename} ({file_size_mb:.1f} MB)")
            data = load_small_json(filepath)
            print(f"      {len(data)} 条")
            all_data.extend(data)
    
    return all_data


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


def main():
    parser = argparse.ArgumentParser(description="数据集混合脚本")
    parser.add_argument("--proprietary_ratio", type=float, default=0.65,
                        help="专有数据集比例 (默认: 0.65)")
    parser.add_argument("--general_ratio", type=float, default=0.25,
                        help="通用数据集比例 (默认: 0.25)")
    parser.add_argument("--creative_ratio", type=float, default=0.10,
                        help="创意写作数据集比例 (默认: 0.10)")
    parser.add_argument("--total_samples", type=int, default=300000,
                        help="最终数据集总条数 (默认: 300000，建议 20-50万)")
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
    
    total = args.total_samples
    proprietary_count = int(total * args.proprietary_ratio)
    general_count = int(total * args.general_ratio)
    creative_count = int(total * args.creative_ratio)
    
    print("========================================")
    print("  数据集混合")
    print("========================================")
    print(f"  专有数据比例: {args.proprietary_ratio:.0%} → {proprietary_count} 条")
    print(f"  通用数据比例: {args.general_ratio:.0%} → {general_count} 条")
    print(f"  创意写作比例: {args.creative_ratio:.0%} → {creative_count} 条")
    print(f"  总计: {total} 条")
    print(f"")
    print(f"  专有数据路径: {PROPRIETARY_DIR}")
    print(f"  通用数据路径: {GENERAL_DIR}")
    print(f"  输出路径: {OUTPUT_DIR}")
    print("")
    
    # ---- 加载专有数据集（流式采样，不全部加载） ----
    print("[1/3] 采样专有数据集（流式）...")
    proprietary_file = os.path.join(PROPRIETARY_DIR, "proprietary_nsfw.json")
    if not os.path.exists(proprietary_file):
        print(f"  错误: 专有数据集不存在: {proprietary_file}")
        print(f"  请先运行 process_proprietary_datasets.py")
        return
    
    file_size_gb = os.path.getsize(proprietary_file) / 1024**3
    print(f"    文件大小: {file_size_gb:.1f} GB")
    
    proprietary_data = stream_sample_large_json(
        proprietary_file, proprietary_count, seed=args.seed
    )
    print(f"  ✓ 专有数据采样完成: {len(proprietary_data)} 条\n")
    
    # ---- 加载通用数据集（文件小，直接加载） ----
    print("[2/3] 加载通用数据集...")
    general_data = collect_small_json_files(GENERAL_DIR)
    print(f"  总计: {len(general_data)} 条")
    
    if len(general_data) == 0:
        print("  警告: 通用数据集为空，将全部使用专有数据")
        general_count = 0
    
    # 如果没有创意写作数据，将其比例分配给通用数据
    print("\n[3/3] 加载创意写作数据集...")
    creative_dir = os.path.join("/root/autodl-tmp/datasets", "creative", "converted")
    creative_data = collect_small_json_files(creative_dir)
    print(f"  总计: {len(creative_data)} 条")
    
    if len(creative_data) == 0 and creative_count > 0:
        print("  注意: 无创意写作数据，将比例分配给通用数据")
        general_count += creative_count
        creative_count = 0
    
    # ---- 采样通用和创意数据 ----
    print(f"\n[采样]")
    print(f"  专有数据: {len(proprietary_data)} 条 (已完成)")
    
    if general_count > 0 and len(general_data) > 0:
        general_sampled = sample_dataset(general_data, general_count)
        print(f"  通用数据: {len(general_sampled)} 条")
    else:
        general_sampled = []
        print(f"  通用数据: 0 条 (跳过)")
    
    if creative_count > 0 and len(creative_data) > 0:
        creative_sampled = sample_dataset(creative_data, creative_count)
        print(f"  创意写作: {len(creative_sampled)} 条")
    else:
        creative_sampled = []
    
    # ---- 合并并打乱 ----
    print(f"\n[合并] 合并所有数据...")
    final_data = proprietary_data + general_sampled + creative_sampled
    random.shuffle(final_data)
    print(f"  最终数据集: {len(final_data)} 条")
    
    # ---- 保存 ----
    print(f"\n[保存] 写入文件...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "train_mixed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    file_size_mb = os.path.getsize(output_path) / 1024**2
    
    print(f"\n========================================")
    print(f"  混合完成!")
    print(f"========================================")
    print(f"  最终数据集: {len(final_data)} 条")
    print(f"  输出路径: {output_path}")
    print(f"  文件大小: {file_size_mb:.1f} MB")
    print(f"")
    print(f"  组成:")
    print(f"    专有数据: {len(proprietary_data)} 条 ({len(proprietary_data)/len(final_data)*100:.1f}%)")
    print(f"    通用数据: {len(general_sampled)} 条 ({len(general_sampled)/len(final_data)*100:.1f}%)")
    if creative_sampled:
        print(f"    创意写作: {len(creative_sampled)} 条 ({len(creative_sampled)/len(final_data)*100:.1f}%)")
    print(f"========================================")
    
    # ---- 生成数据集信息文件 ----
    info = {
        "total_samples": len(final_data),
        "proprietary_count": len(proprietary_data),
        "general_count": len(general_sampled),
        "creative_count": len(creative_sampled),
        "proprietary_ratio": args.proprietary_ratio,
        "general_ratio": args.general_ratio,
        "creative_ratio": args.creative_ratio,
        "seed": args.seed,
        "output_path": output_path,
    }
    info_path = os.path.join(OUTPUT_DIR, "mix_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  数据集信息: {info_path}")


if __name__ == "__main__":
    main()
