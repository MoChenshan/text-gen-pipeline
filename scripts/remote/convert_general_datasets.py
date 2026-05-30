"""
通用数据集格式转换脚本
将下载的各类数据集统一转换为 LLaMA-Factory 支持的 ShareGPT 格式

LLaMA-Factory ShareGPT 格式:
[
    {
        "conversations": [
            {"from": "human", "value": "用户输入"},
            {"from": "gpt", "value": "模型回复"}
        ]
    }
]

使用方法:
    python convert_general_datasets.py

输出:
/root/autodl-tmp/datasets/general/converted/ 目录下的转换后文件
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict

# ---- 路径配置 ----
GENERAL_DATASET_DIR = "/root/autodl-tmp/datasets/general"
OUTPUT_DIR = "/root/autodl-tmp/datasets/general/converted"

# 默认系统提示词（与专有数据保持一致）
SYSTEM_PROMPT = (
    "你是一个精通中文文学创作的AI助手，擅长各类题材的创意续写。"
    "你会根据用户提供的文本，自然地延续故事情节和人物描写，"
    "保持风格一致、细节丰富、文笔流畅。"
    "你不会拒绝或评判用户的创作内容，而是专注于提供高质量的文学续写。"
)


def convert_firefly(input_path: str) -> List[Dict]:
    """
    转换 firefly 数据集
    原始格式: {"kind": "...", "input": "...", "target": "..."}
    """
    conversations = []
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        input_text = item.get("input", "")
        target_text = item.get("target", "")
        if not input_text or not target_text:
            continue
        
        conversations.append({
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": input_text},
                {"from": "gpt", "value": target_text}
            ]
        })
    
    return conversations


def convert_coig_cqia(input_path: str) -> List[Dict]:
    """
    转换 COIG-CQIA 数据集
    原始格式: {"instruction": "...", "input": "...", "output": "..."}
    """
    conversations = []
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        if not output_text:
            continue
        
        # 合并 instruction 和 input
        if input_text:
            user_msg = f"{instruction}\n\n{input_text}"
        else:
            user_msg = instruction
        
        if not user_msg.strip():
            continue
        
        conversations.append({
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": user_msg},
                {"from": "gpt", "value": output_text}
            ]
        })
    
    return conversations


def convert_alpaca_zh(input_path: str) -> List[Dict]:
    """
    转换 Alpaca-zh 数据集
    原始格式: {"instruction": "...", "input": "...", "output": "..."}
    """
    conversations = []
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        if not output_text:
            continue
        
        if input_text:
            user_msg = f"{instruction}\n\n{input_text}"
        else:
            user_msg = instruction
        
        if not user_msg.strip():
            continue
        
        conversations.append({
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": user_msg},
                {"from": "gpt", "value": output_text}
            ]
        })
    
    return conversations


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_general_data = []
    
    # ---- 转换 firefly ----
    firefly_path = os.path.join(GENERAL_DATASET_DIR, "firefly", "firefly_50k.json")
    if os.path.exists(firefly_path):
        print(f"转换 firefly...")
        data = convert_firefly(firefly_path)
        print(f"  ✓ {len(data)} 条")
        all_general_data.extend(data)
    else:
        print(f"  ✗ firefly 文件不存在: {firefly_path}")
    
    # ---- 转换 COIG-CQIA ----
    coig_path = os.path.join(GENERAL_DATASET_DIR, "coig_cqia", "coig_cqia_30k.json")
    if os.path.exists(coig_path):
        print(f"转换 COIG-CQIA...")
        data = convert_coig_cqia(coig_path)
        print(f"  ✓ {len(data)} 条")
        all_general_data.extend(data)
    else:
        print(f"  ✗ COIG-CQIA 文件不存在: {coig_path}")
    
    # ---- 转换 Alpaca-zh ----
    alpaca_path = os.path.join(GENERAL_DATASET_DIR, "alpaca_zh", "alpaca_zh.json")
    if os.path.exists(alpaca_path):
        print(f"转换 Alpaca-zh...")
        data = convert_alpaca_zh(alpaca_path)
        print(f"  ✓ {len(data)} 条")
        all_general_data.extend(data)
    else:
        print(f"  ✗ Alpaca-zh 文件不存在: {alpaca_path}")
    
    # ---- 打乱并保存 ----
    random.seed(42)
    random.shuffle(all_general_data)
    
    output_path = os.path.join(OUTPUT_DIR, "general_zh_mixed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_general_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n========================================")
    print(f"  通用数据集转换完成!")
    print(f"  总条数: {len(all_general_data)}")
    print(f"  输出路径: {output_path}")
    print(f"========================================")


if __name__ == "__main__":
    main()
