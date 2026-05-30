"""
专有数据集下载与预处理脚本
=========================

数据源:
  1. ystemsrx/Erotic_Literature_Collection - 多个分类JSON文件，字段: {"text": "..."}
  2. a686d380/h-corpus-2023 - zip压缩包，内含中文小说文本
  3. Seikaijyu/Sex-novel-filtered - JSONL文件 (SexNovel.jsonl)

处理流程:
  1. 从 HuggingFace 下载原始数据
  2. 解析各数据集格式，提取纯文本
  3. 将长文本切分为"续写对"（prefix → continuation）
  4. 转换为 LLaMA-Factory ShareGPT 格式
5. 输出到 /root/autodl-tmp/datasets/proprietary_converted/

使用方法（在远端 AutoDL 执行）:
    python process_proprietary_datasets.py [--max_samples 50000] [--chunk_size 2048]

输出格式 (ShareGPT):
    [
        {
            "conversations": [
                {"from": "human", "value": "请续写以下内容：\n{prefix}"},
                {"from": "gpt", "value": "{continuation}"}
            ]
        }
    ]
"""

import os
import sys
import json
import random
import zipfile
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ---- 路径配置 ----
DATASET_DIR = "/root/autodl-tmp/datasets/proprietary"
RAW_DIR = os.path.join(DATASET_DIR, "raw")
# 输出到数据盘 autodl-tmp，避免 autodl-fs 网盘容量限制
OUTPUT_DIR = "/root/autodl-tmp/datasets/proprietary_converted"

# HuggingFace 镜像（国内加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 将 HF 缓存目录指向数据盘，避免撑满系统盘
os.environ["HF_HOME"] = "/root/autodl-tmp/cache/huggingface"


# ============================================================
# 第一部分：数据下载
# ============================================================

def _check_h_corpus_extracted(ds2_dir: str) -> bool:
    """检查 h-corpus-2023 是否已经成功解压（有实际文本文件）"""
    for root, dirs, files in os.walk(ds2_dir):
        for f in files:
            if f.endswith(('.txt', '.json', '.jsonl', '.csv')):
                # 找到至少一个文本文件，说明已解压
                return True
    return False


def _download_with_aria2c(url: str, output_dir: str, filename: str) -> str:
    """使用 aria2c 多线程下载（速度远快于 Python 单线程）"""
    import subprocess
    
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path):
        return output_path
    
    # 检查 aria2c 是否可用
    try:
        subprocess.run(["aria2c", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None  # aria2c 不可用，回退到 Python 下载
    
    print(f"  使用 aria2c 多线程下载: {filename}")
    cmd = [
        "aria2c",
        "-x", "16",           # 16 线程
        "-s", "16",           # 16 分片
        "-k", "10M",          # 每片最小 10MB
        "--dir", output_dir,
        "--out", filename,
        "--continue=true",    # 支持断点续传
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0 and os.path.exists(output_path):
        return output_path
    return None


def download_datasets(skip_h_corpus: bool = False):
    """
    从 HuggingFace 下载专有数据集
    
    Args:
        skip_h_corpus: 是否跳过 h-corpus-2023（7.18GB，下载较慢）
    """
    from huggingface_hub import snapshot_download, hf_hub_download
    
    # 尝试启用 hf-transfer 加速（如果已安装）
    try:
        import hf_transfer
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("  ✓ hf-transfer 加速已启用")
    except ImportError:
        print("  ℹ hf-transfer 未安装，使用默认下载（可选: pip install hf-transfer）")
    
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # ---- 数据集 1: Erotic_Literature_Collection (约2GB，多个小JSON) ----
    ds1_dir = os.path.join(RAW_DIR, "erotic_literature_collection")
    if not os.path.exists(ds1_dir) or len(os.listdir(ds1_dir)) < 5:
        print("[1/3] 下载 Erotic_Literature_Collection...")
        snapshot_download(
            "ystemsrx/Erotic_Literature_Collection",
            repo_type="dataset",
            local_dir=ds1_dir,
            ignore_patterns=["*.md", ".gitattributes"],
        )
        print(f"  ✓ 完成: {ds1_dir}")
    else:
        print(f"[1/3] Erotic_Literature_Collection 已存在，跳过")
    
    # ---- 数据集 2: h-corpus-2023 (7.18GB zip，可选) ----
    ds2_dir = os.path.join(RAW_DIR, "h-corpus-2023")
    if skip_h_corpus:
        print(f"[2/3] h-corpus-2023 已跳过（--skip_h_corpus）")
    elif os.path.exists(ds2_dir) and _check_h_corpus_extracted(ds2_dir):
        print(f"[2/3] h-corpus-2023 已解压，跳过")
    else:
        print("[2/3] 下载 h-corpus-2023 (7.18GB，较大)...")
        os.makedirs(ds2_dir, exist_ok=True)
        
        # 优先尝试 aria2c 多线程下载
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        download_url = f"{hf_endpoint}/datasets/a686d380/h-corpus-2023/resolve/main/h-corpus.zip"
        zip_path = _download_with_aria2c(download_url, ds2_dir, "h-corpus.zip")
        
        if zip_path is None:
            # 回退到 huggingface_hub 下载
            print("  aria2c 不可用，使用 Python 下载（较慢）...")
            zip_path = hf_hub_download(
                "a686d380/h-corpus-2023",
                "h-corpus.zip",
                repo_type="dataset",
                local_dir=ds2_dir,
            )
        
        # 解压
        print("  解压 h-corpus.zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ds2_dir)
        # 解压完成后删除 zip 节省空间
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("  已删除 zip 文件节省空间")
        print(f"  ✓ 完成: {ds2_dir}")
    
    # ---- 数据集 3: Sex-novel-filtered (较小，JSONL) ----
    ds3_dir = os.path.join(RAW_DIR, "sex-novel-filtered")
    if not os.path.exists(ds3_dir) or not os.listdir(ds3_dir):
        print("[3/3] 下载 Sex-novel-filtered...")
        os.makedirs(ds3_dir, exist_ok=True)
        hf_hub_download(
            "Seikaijyu/Sex-novel-filtered",
            "SexNovel.jsonl",
            repo_type="dataset",
            local_dir=ds3_dir,
        )
        print(f"  ✓ 完成: {ds3_dir}")
    else:
        print(f"[3/3] Sex-novel-filtered 已存在，跳过")
    
    print("\n数据集下载完成!")


# ============================================================
# 第二部分：文本提取
# ============================================================

def extract_texts_from_erotic_literature(ds_dir: str) -> List[str]:
    """
    从 Erotic_Literature_Collection 提取文本
    格式: 多个 JSON 文件，每个文件是一个列表，每项有 "text" 字段
    """
    texts = []
    json_files = [f for f in os.listdir(ds_dir) if f.endswith(".json")]
    
    print(f"  发现 {len(json_files)} 个 JSON 文件")
    for filename in json_files:
        filepath = os.path.join(ds_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # 列表格式: [{"text": "..."}, ...]
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"].strip()
                        if len(text) > 200:  # 过滤太短的文本
                            texts.append(text)
                    elif isinstance(item, str):
                        if len(item.strip()) > 200:
                            texts.append(item.strip())
            elif isinstance(data, dict):
                # 可能是 {"text": "..."} 或其他格式
                if "text" in data:
                    texts.append(data["text"].strip())
            
            print(f"    {filename}: 提取 {len(data) if isinstance(data, list) else 1} 条")
        except Exception as e:
            print(f"    {filename}: 读取失败 - {e}")
            # 尝试其他编码
            try:
                with open(filepath, "r", encoding="gbk") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "text" in item:
                            text = item["text"].strip()
                            if len(text) > 200:
                                texts.append(text)
                print(f"    {filename}: (GBK编码) 提取成功")
            except:
                print(f"    {filename}: GBK编码也失败，跳过")
    
    return texts


def extract_texts_from_h_corpus(ds_dir: str) -> List[str]:
    """
    从 h-corpus-2023 提取文本
    格式: zip 解压后可能是 txt/json 文件
    """
    texts = []
    
    # 递归查找所有文本文件
    for root, dirs, files in os.walk(ds_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            if filename.endswith(".txt"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if len(text) > 200:
                        texts.append(text)
                except UnicodeDecodeError:
                    try:
                        with open(filepath, "r", encoding="gbk") as f:
                            text = f.read().strip()
                        if len(text) > 200:
                            texts.append(text)
                    except:
                        pass
            
            elif filename.endswith(".json"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                # 尝试常见字段名
                                text = item.get("text") or item.get("content") or item.get("story") or ""
                                if len(text.strip()) > 200:
                                    texts.append(text.strip())
                            elif isinstance(item, str) and len(item.strip()) > 200:
                                texts.append(item.strip())
                    elif isinstance(data, dict):
                        text = data.get("text") or data.get("content") or ""
                        if len(text.strip()) > 200:
                            texts.append(text.strip())
                except:
                    pass
            
            elif filename.endswith(".jsonl"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            item = json.loads(line)
                            if isinstance(item, dict):
                                text = item.get("text") or item.get("content") or item.get("story") or ""
                                if len(text.strip()) > 200:
                                    texts.append(text.strip())
                except:
                    pass
    
    print(f"  从 h-corpus-2023 提取 {len(texts)} 篇文本")
    return texts


def extract_texts_from_sex_novel(ds_dir: str) -> List[str]:
    """
    从 Sex-novel-filtered 提取文本
    格式: SexNovel.jsonl，每行一个 JSON 对象
    """
    texts = []
    jsonl_path = os.path.join(ds_dir, "SexNovel.jsonl")
    
    if not os.path.exists(jsonl_path):
        # 查找任何 jsonl 文件
        for f in os.listdir(ds_dir):
            if f.endswith(".jsonl"):
                jsonl_path = os.path.join(ds_dir, f)
                break
    
    if not os.path.exists(jsonl_path):
        print(f"  警告: 未找到 JSONL 文件: {ds_dir}")
        return texts
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    # 尝试常见字段名
                    text = (item.get("text") or item.get("content") or 
                            item.get("story") or item.get("novel") or "")
                    if len(text.strip()) > 200:
                        texts.append(text.strip())
                elif isinstance(item, str) and len(item.strip()) > 200:
                    texts.append(item.strip())
            except json.JSONDecodeError:
                continue
    
    print(f"  从 Sex-novel-filtered 提取 {len(texts)} 篇文本")
    return texts


# ============================================================
# 第三部分：文本清洗
# ============================================================

def clean_text(text: str) -> str:
    """清洗文本，去除无关内容"""
    # 去除多余空行（保留段落结构）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 去除行首行尾空格
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # 去除常见的网站水印/广告文本
    watermarks = [
        r'本文来自.*?网',
        r'更多精彩.*?请访问',
        r'www\..*?\.com',
        r'http[s]?://\S+',
        r'【.*?小说网.*?】',
        r'手机用户请到.*?阅读',
        r'本站.*?地址',
    ]
    for pattern in watermarks:
        text = re.sub(pattern, '', text)
    
    # 去除多余空格
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n ', '\n', text)
    
    return text.strip()


# ============================================================
# 第四部分：构建训练对
# ============================================================

def split_into_paragraphs(text: str) -> List[str]:
    """将文本按段落分割"""
    # 按换行分割，合并短段落
    raw_paragraphs = text.split('\n')
    paragraphs = []
    current = ""
    
    for p in raw_paragraphs:
        p = p.strip()
        if not p:
            if current:
                paragraphs.append(current)
                current = ""
            continue
        
        if current:
            current += "\n" + p
        else:
            current = p
        
        # 如果当前段落足够长，保存
        if len(current) > 300:
            paragraphs.append(current)
            current = ""
    
    if current:
        paragraphs.append(current)
    
    return paragraphs


def build_continuation_pairs(
    text: str,
    min_prefix_len: int = 256,
    max_prefix_len: int = 1536,
    min_continuation_len: int = 256,
    max_continuation_len: int = 2048,
    stride: int = 512,
) -> List[Tuple[str, str]]:
    """
    从长文本中构建续写对 (prefix, continuation)
    
    策略:
    - 使用滑动窗口从文本中截取片段
    - prefix: 前文内容（作为用户输入）
    - continuation: 续写内容（作为模型输出）
    - 尽量在段落/句子边界切分
    """
    pairs = []
    text_len = len(text)
    
    if text_len < min_prefix_len + min_continuation_len:
        return pairs
    
    # 使用步进式滑动窗口，直接按 stride 间隔选取候选切分位置
    # 避免遍历每个字符和 O(n²) 的间距检查
    cursor = min_prefix_len
    
    while cursor < text_len - min_continuation_len:
        # 在 cursor 附近寻找最近的句子边界（向前搜索一小段范围）
        best_point = None
        search_range = min(200, text_len - cursor)  # 在200字符范围内找句子边界
        
        for offset in range(search_range):
            pos = cursor + offset
            if pos >= text_len - min_continuation_len:
                break
            if text[pos] in '。！？\n':
                best_point = pos + 1
                break
        
        # 如果附近没有句子边界，就用当前位置
        if best_point is None:
            best_point = cursor
        
        # 确定 prefix 范围
        prefix_start = max(0, best_point - max_prefix_len)
        # 尝试在段落开头开始
        newline_pos = text.find('\n', prefix_start, prefix_start + 100)
        if newline_pos != -1:
            prefix_start = newline_pos + 1
        
        prefix = text[prefix_start:best_point].strip()
        
        # 确定 continuation 范围
        continuation_end = min(text_len, best_point + max_continuation_len)
        # 尝试在句子结尾结束
        for end_char in ['。', '！', '？', '\n']:
            last_end = text.rfind(end_char, best_point, continuation_end)
            if last_end > best_point + min_continuation_len:
                continuation_end = last_end + 1
                break
        
        continuation = text[best_point:continuation_end].strip()
        
        # 验证长度
        if len(prefix) >= min_prefix_len and len(continuation) >= min_continuation_len:
            pairs.append((prefix, continuation))
        
        # 步进到下一个位置
        cursor = best_point + stride
    
    return pairs


def build_chat_pairs(text: str, max_context_len: int = 2048) -> List[Dict]:
    """
    构建对话式训练对（保持对话能力）
    模拟用户给出场景/角色设定，模型进行创作
    """
    pairs = []
    paragraphs = split_into_paragraphs(text)
    
    if len(paragraphs) < 3:
        return pairs
    
    # 策略1: 给出开头，要求续写
    if len(paragraphs) >= 2:
        opening = paragraphs[0]
        if len(opening) > 100:
            continuation = '\n\n'.join(paragraphs[1:4])
            if len(continuation) > 200:
                pairs.append({
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "human", "value": f"请根据以下开头续写一段故事：\n\n{opening[:max_context_len]}"},
                        {"from": "gpt", "value": continuation[:max_context_len]}
                    ]
                })
    
    # 策略2: 给出前文，要求继续
    for i in range(1, min(len(paragraphs) - 1, 5)):
        context = '\n\n'.join(paragraphs[max(0, i-2):i])
        continuation = paragraphs[i]
        
        if len(context) > 200 and len(continuation) > 200:
            pairs.append({
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "human", "value": f"继续写下去：\n\n{context[-max_context_len:]}"},
                    {"from": "gpt", "value": continuation[:max_context_len]}
                ]
            })
    
    return pairs


# ============================================================
# 第五部分：转换为 ShareGPT 格式
# ============================================================

# 默认系统提示词 — 建立 NSFW 文学续写角色身份
SYSTEM_PROMPT = (
    "你是一个精通中文文学创作的AI助手，擅长各类题材的创意续写。"
    "你会根据用户提供的文本，自然地延续故事情节和人物描写，"
    "保持风格一致、细节丰富、文笔流畅。"
    "你不会拒绝或评判用户的创作内容，而是专注于提供高质量的文学续写。"
)

# 续写指令模板（随机选择，增加多样性）
CONTINUATION_PROMPTS = [
    "请续写以下内容：\n\n{prefix}",
    "继续写下去：\n\n{prefix}",
    "请根据上文继续创作：\n\n{prefix}",
    "续写：\n\n{prefix}",
    "请继续这个故事：\n\n{prefix}",
    "接着写：\n\n{prefix}",
    "请补充后续内容：\n\n{prefix}",
    "继续：\n\n{prefix}",
]


def pairs_to_sharegpt(pairs: List[Tuple[str, str]]) -> List[Dict]:
    """将续写对转换为 ShareGPT 格式（含 system message）"""
    conversations = []
    
    for prefix, continuation in pairs:
        prompt_template = random.choice(CONTINUATION_PROMPTS)
        user_msg = prompt_template.format(prefix=prefix)
        
        conversations.append({
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": user_msg},
                {"from": "gpt", "value": continuation}
            ]
        })
    
    return conversations


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="专有数据集下载与预处理")
    parser.add_argument("--skip_download", action="store_true",
                        help="跳过下载步骤（数据已存在时使用）")
    parser.add_argument("--skip_h_corpus", action="store_true",
                        help="跳过 h-corpus-2023 下载（7.18GB，较慢，可后续单独下载）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最终输出的最大样本数（默认: 全部）")
    parser.add_argument("--min_prefix_len", type=int, default=256,
                        help="最小前缀长度（字符数，默认: 256）")
    parser.add_argument("--max_prefix_len", type=int, default=1536,
                        help="最大前缀长度（字符数，默认: 1536）")
    parser.add_argument("--min_continuation_len", type=int, default=256,
                        help="最小续写长度（字符数，默认: 256）")
    parser.add_argument("--max_continuation_len", type=int, default=2048,
                        help="最大续写长度（字符数，默认: 2048）")
    parser.add_argument("--stride", type=int, default=512,
                        help="滑动窗口步长（字符数，默认: 512）")
    parser.add_argument("--include_chat", action="store_true", default=True,
                        help="是否包含对话式训练对（默认: True）")
    parser.add_argument("--chat_ratio", type=float, default=0.2,
                        help="对话式训练对占比（默认: 0.2）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认: 42）")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 60)
    print("  专有数据集下载与预处理")
    print("=" * 60)
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  前缀长度: {args.min_prefix_len} - {args.max_prefix_len}")
    print(f"  续写长度: {args.min_continuation_len} - {args.max_continuation_len}")
    print(f"  滑动步长: {args.stride}")
    print(f"  模式: 流式处理（低内存占用）")
    print("")
    
    # ---- Step 1: 下载 ----
    if not args.skip_download:
        print("[Step 1] 下载数据集...")
        download_datasets(skip_h_corpus=args.skip_h_corpus)
    else:
        print("[Step 1] 跳过下载")
    
    # ---- Step 2 & 3: 提取 + 清洗 + 构建训练对（流式处理）----
    # 不再全部加载到内存，而是逐数据集处理，每篇文本处理完立即写入文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "proprietary_nsfw.jsonl")
    
    print("\n[Step 2-4] 流式提取+清洗+构建训练对...")
    
    total_continuation_count = 0
    total_chat_count = 0
    total_source_texts = 0
    total_cleaned_texts = 0
    
    # 数据集列表
    datasets = []
    ds1_dir = os.path.join(RAW_DIR, "erotic_literature_collection")
    if os.path.exists(ds1_dir):
        datasets.append(("Erotic_Literature_Collection", ds1_dir, "erotic"))
    ds2_dir = os.path.join(RAW_DIR, "h-corpus-2023")
    if os.path.exists(ds2_dir):
        datasets.append(("h-corpus-2023", ds2_dir, "h_corpus"))
    ds3_dir = os.path.join(RAW_DIR, "sex-novel-filtered")
    if os.path.exists(ds3_dir):
        datasets.append(("Sex-novel-filtered", ds3_dir, "sex_novel"))
    
    if not datasets:
        print("错误: 未找到任何数据集！请检查数据集路径。")
        sys.exit(1)
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        for ds_name, ds_dir, ds_type in datasets:
            print(f"\n  处理 {ds_name}...")
            
            # 提取文本
            if ds_type == "erotic":
                raw_texts = extract_texts_from_erotic_literature(ds_dir)
            elif ds_type == "h_corpus":
                raw_texts = extract_texts_from_h_corpus(ds_dir)
            else:
                raw_texts = extract_texts_from_sex_novel(ds_dir)
            
            total_source_texts += len(raw_texts)
            print(f"    提取 {len(raw_texts)} 篇文本")
            
            ds_continuation_count = 0
            ds_chat_count = 0
            ds_cleaned = 0
            
            for i, text in enumerate(raw_texts):
                if (i + 1) % 1000 == 0:
                    print(f"    进度: {i+1}/{len(raw_texts)} | "
                          f"续写对: {ds_continuation_count} | 对话对: {ds_chat_count}")
                
                # 清洗
                cleaned = clean_text(text)
                del text  # 立即释放原始文本
                if len(cleaned) <= 500:
                    continue
                ds_cleaned += 1
                
                # 续写对
                pairs = build_continuation_pairs(
                    cleaned,
                    min_prefix_len=args.min_prefix_len,
                    max_prefix_len=args.max_prefix_len,
                    min_continuation_len=args.min_continuation_len,
                    max_continuation_len=args.max_continuation_len,
                    stride=args.stride,
                )
                
                # 直接转换为 ShareGPT 并写入文件
                for prefix, continuation in pairs:
                    prompt_template = random.choice(CONTINUATION_PROMPTS)
                    user_msg = prompt_template.format(prefix=prefix)
                    record = {
                        "conversations": [
                            {"from": "system", "value": SYSTEM_PROMPT},
                            {"from": "human", "value": user_msg},
                            {"from": "gpt", "value": continuation}
                        ]
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    ds_continuation_count += 1
                
                # 对话对
                if args.include_chat:
                    chat_pairs = build_chat_pairs(cleaned)
                    for record in chat_pairs:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        ds_chat_count += 1
                
                del cleaned  # 释放清洗后文本
                del pairs
            
            total_continuation_count += ds_continuation_count
            total_chat_count += ds_chat_count
            total_cleaned_texts += ds_cleaned
            print(f"    完成: {ds_cleaned} 篇有效文本, "
                  f"{ds_continuation_count} 续写对, {ds_chat_count} 对话对")
    
    print(f"\n  总计: {total_source_texts} 篇原始文本 → "
          f"{total_cleaned_texts} 篇有效 → "
          f"{total_continuation_count} 续写对 + {total_chat_count} 对话对")
    
    # ---- Step 5: 对话对比例调整 & 采样 ----
    # 如果对话对过多需要采样，或者需要限制总数
    # 使用两遍扫描法处理 JSONL（内存友好）
    if args.include_chat and total_chat_count > 0:
        target_chat_count = int(total_continuation_count * args.chat_ratio / (1 - args.chat_ratio))
        if total_chat_count > target_chat_count:
            print(f"\n[Step 5] 调整对话对比例: {total_chat_count} → {target_chat_count}")
            _resample_chat_pairs(output_path, target_chat_count, total_chat_count)
            total_chat_count = target_chat_count
    
    if args.max_samples:
        total_all = total_continuation_count + total_chat_count
        if total_all > args.max_samples:
            print(f"\n[Step 5] 采样: {total_all} → {args.max_samples}")
            _random_sample_jsonl(output_path, args.max_samples)
            total_continuation_count = -1  # 采样后不再区分
            total_chat_count = -1
    
    # ---- Step 6: 打乱 JSONL 行序 ----
    print("\n[Step 6] 打乱数据顺序...")
    _shuffle_jsonl(output_path)
    
    # ---- Step 7: 保存统计信息 ----
    # 统计最终行数
    final_count = 0
    with open(output_path, "r", encoding="utf-8") as f:
        for _ in f:
            final_count += 1
    
    stats = {
        "total_samples": final_count,
        "continuation_pairs": total_continuation_count if total_continuation_count >= 0 else "sampled",
        "chat_pairs": total_chat_count if total_chat_count >= 0 else "sampled",
        "source_texts": total_source_texts,
        "cleaned_texts": total_cleaned_texts,
        "params": {
            "min_prefix_len": args.min_prefix_len,
            "max_prefix_len": args.max_prefix_len,
            "min_continuation_len": args.min_continuation_len,
            "max_continuation_len": args.max_continuation_len,
            "stride": args.stride,
            "chat_ratio": args.chat_ratio,
        }
    }
    stats_path = os.path.join(OUTPUT_DIR, "proprietary_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n{'=' * 60}")
    print(f"  处理完成!")
    print(f"{'=' * 60}")
    print(f"  输出文件: {output_path}")
    print(f"  统计信息: {stats_path}")
    print(f"  总样本数: {final_count}")
    print(f"  文件大小: {file_size_mb:.1f} MB")
    print(f"\n  下一步: 运行 mix_datasets.py 混合通用数据集")


def _resample_chat_pairs(jsonl_path: str, target_count: int, total_chat_count: int):
    """
    对 JSONL 中的对话对进行下采样（保留所有续写对）
    使用两遍扫描避免全部加载到内存
    """
    # Pass 1: 找出所有对话对的行号
    chat_line_indices = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line)
                convs = record.get("conversations", [])
                # 对话对的 human 消息不以续写指令开头
                if convs and len(convs) >= 2:
                    human_msg = convs[1].get("value", "")
                    # 续写对的 human 消息包含 {prefix} 占位内容（较长）
                    # 对话对更短，且以特定指令开头
                    for prompt_prefix in ["请根据以下开头续写", "继续写下去：\n\n"] + CONTINUATION_PROMPTS:
                        if human_msg.startswith(prompt_prefix.split("{")[0]):
                            break
                    else:
                        chat_line_indices.append(i)
            except:
                pass
    
    if len(chat_line_indices) <= target_count:
        return
    
    # 随机选择要保留的对话对行号
    keep_indices = set(random.sample(chat_line_indices, target_count))
    remove_indices = set(chat_line_indices) - keep_indices
    
    # Pass 2: 重写文件，跳过被移除的对话对
    tmp_path = jsonl_path + ".tmp"
    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i not in remove_indices:
                fout.write(line)
    
    os.replace(tmp_path, jsonl_path)
    print(f"    对话对: {total_chat_count} → {target_count}")


def _random_sample_jsonl(jsonl_path: str, target_count: int):
    """从 JSONL 文件中随机采样指定行数"""
    import tempfile
    
    # Pass 1: 统计总行数
    total_lines = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1
    
    if total_lines <= target_count:
        return
    
    # 随机选择要保留的行号
    keep_indices = set(random.sample(range(total_lines), target_count))
    
    # Pass 2: 只写入选中的行
    tmp_path = jsonl_path + ".tmp"
    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i in keep_indices:
                fout.write(line)
    
    os.replace(tmp_path, jsonl_path)


def _shuffle_jsonl(jsonl_path: str):
    """打乱 JSONL 文件行序（使用临时文件，内存友好）"""
    # 读取所有行的偏移量
    offsets = []
    with open(jsonl_path, "rb") as f:
        offset = 0
        for line in f:
            offsets.append(offset)
            offset += len(line)
    
    # 随机打乱偏移量
    random.shuffle(offsets)
    
    # 按打乱后的顺序重写文件
    tmp_path = jsonl_path + ".shuffled"
    with open(jsonl_path, "rb") as fin, \
         open(tmp_path, "wb") as fout:
        for offset in offsets:
            fin.seek(offset)
            line = fin.readline()
            fout.write(line)
    
    os.replace(tmp_path, jsonl_path)


if __name__ == "__main__":
    main()
