"""
LoRA 权重合并脚本
将训练好的 LoRA 适配器合并到基座模型中，生成完整模型

使用方法:
    python merge_lora.py [--lora_path PATH] [--output_path PATH]
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

# LLaMA-Factory CLI 默认路径（conda 环境）
LLAMAFACTORY_CLI_DEFAULT = "/root/miniconda3/envs/llama_factory/bin/llamafactory-cli"


def find_llamafactory_cli() -> str:
    """查找 llamafactory-cli 可执行文件"""
    # 优先使用环境变量
    env_path = os.environ.get("LLAMAFACTORY_CLI")
    if env_path and os.path.isfile(env_path):
        return env_path
    
    # 尝试 PATH 中查找
    cli_in_path = shutil.which("llamafactory-cli")
    if cli_in_path:
        return cli_in_path
    
    # 使用默认 conda 环境路径
    if os.path.isfile(LLAMAFACTORY_CLI_DEFAULT):
        return LLAMAFACTORY_CLI_DEFAULT
    
    return None


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    export_dtype: str = "bf16",
):
    """
    使用 LLaMA-Factory 的导出功能合并 LoRA
    """
    import json
    import tempfile
    import yaml
    
    # 查找 llamafactory-cli
    cli_path = find_llamafactory_cli()
    if cli_path is None:
        print("错误: 找不到 llamafactory-cli!")
        print("请确保已激活 conda 环境: conda activate llama_factory")
        print(f"或设置环境变量: export LLAMAFACTORY_CLI=/path/to/llamafactory-cli")
        sys.exit(1)
    
    print(f"使用 CLI: {cli_path}")
    
    # 构建 LLaMA-Factory 导出配置
    export_config = {
        "model_name_or_path": base_model_path,
        "adapter_name_or_path": lora_path,
        "template": "qwen3",
        "finetuning_type": "lora",
        "export_dir": output_path,
        "export_size": 5,  # 每个分片最大 5GB
        "export_device": "auto",
        "export_legacy_format": False,
    }
    
    # 设置导出精度（export 命令使用 export_dtype 参数）
    export_config["export_dtype"] = export_dtype
    
    # 写入临时 YAML 配置文件（LLaMA-Factory 对 YAML 解析更稳定）
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(export_config, f, default_flow_style=False, allow_unicode=True)
        config_path = f.name
    
    print(f"合并配置: {json.dumps(export_config, indent=2, ensure_ascii=False)}")
    print(f"临时配置文件: {config_path}")
    
    # 调用 LLaMA-Factory 导出（检查返回码）
    ret = subprocess.run([cli_path, "export", config_path])
    
    # 清理临时文件
    os.unlink(config_path)
    
    if ret.returncode != 0:
        print(f"\n错误: 合并失败! 返回码: {ret.returncode}")
        sys.exit(1)
    
    print(f"\n合并完成! 输出路径: {output_path}")


def merge_lora_manual(
    base_model_path: str,
    lora_path: str,
    output_path: str,
):
    """
    手动合并 LoRA（不依赖 LLaMA-Factory）
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    
    print(f"加载基座模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"加载 LoRA 权重: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("合并 LoRA 到基座模型...")
    model = model.merge_and_unload()
    
    print(f"保存合并后的模型: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, max_shard_size="5GB")
    
    # 复制 tokenizer
    print("复制 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n合并完成! 输出路径: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA 权重合并脚本")
    parser.add_argument("--base_model", type=str,
                        default="/root/autodl-tmp/models/Qwen3.6-27B",
                        help="基座模型路径")
    parser.add_argument("--lora_path", type=str,
default="/root/autodl-tmp/outputs/qwen3.6-27b-lora",
                        help="LoRA 权重路径")
    parser.add_argument("--output_path", type=str,
default="/root/autodl-tmp/outputs/qwen3.6-27b-merged",
                        help="合并后模型输出路径")
    parser.add_argument("--method", type=str, choices=["llamafactory", "manual"],
                        default="llamafactory",
                        help="合并方法: llamafactory (推荐) 或 manual")
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16"],
                        default="bf16",
                        help="导出精度")
    args = parser.parse_args()
    
    print("========================================")
    print("  LoRA 权重合并")
    print("========================================")
    print(f"  基座模型: {args.base_model}")
    print(f"  LoRA 路径: {args.lora_path}")
    print(f"  输出路径: {args.output_path}")
    print(f"  合并方法: {args.method}")
    print(f"  导出精度: {args.dtype}")
    print("")
    
    if args.method == "llamafactory":
        merge_lora(args.base_model, args.lora_path, args.output_path, args.dtype)
    else:
        merge_lora_manual(args.base_model, args.lora_path, args.output_path)


if __name__ == "__main__":
    main()
