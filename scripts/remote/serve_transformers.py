"""
基于 transformers 的 OpenAI 兼容 API 服务
============================================
用途: 当 vLLM/SGLang 不兼容 Qwen3.6 架构时，使用 transformers 原生推理
功能: 提供 /v1/chat/completions、/v1/completions 和 /v1/models 端点
使用: python serve_transformers.py [--model-path PATH] [--port PORT]
"""

import argparse
import time
import uuid
from threading import Thread
from typing import AsyncGenerator, List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

app = FastAPI(title="Qwen3.6 Transformers API Server")

# 全局模型和 tokenizer
model = None
tokenizer = None
MODEL_NAME = "qwen3.6-27b-nsfw"


# ---- 请求/响应模型 ----

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[Message]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 20
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0


class ChatChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


class StreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


# ---- 文本补全请求/响应模型 ----

class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 20
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


# ---- API 端点 ----

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """文本补全接口（续写）"""
    # 直接使用 prompt 作为输入，不套用 chat template
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]

    # 计算 repetition_penalty
    rep_penalty = request.repetition_penalty
    if request.presence_penalty and request.presence_penalty > 0:
        rep_penalty = max(rep_penalty, 1.0 + request.presence_penalty * 0.3)

    # 生成参数
    gen_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature if request.temperature > 0 else 1.0,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "do_sample": request.temperature > 0,
        "repetition_penalty": rep_penalty,
    }

    # 非流式生成
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    new_tokens = outputs[0][prompt_tokens:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    completion_tokens = len(new_tokens)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[
            CompletionChoice(text=response_text)
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """聊天补全接口"""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # 使用 chat template 构建输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # 关闭 thinking 模式，直接输出
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]

    # 计算 repetition_penalty
    rep_penalty = request.repetition_penalty
    if request.presence_penalty and request.presence_penalty > 0:
        rep_penalty = max(rep_penalty, 1.0 + request.presence_penalty * 0.3)

    # 生成参数
    gen_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature if request.temperature > 0 else 1.0,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "do_sample": request.temperature > 0,
        "repetition_penalty": rep_penalty,
    }

    if request.stream:
        return StreamingResponse(
            stream_generate(gen_kwargs, prompt_tokens, request),
            media_type="text/event-stream",
        )

    # 非流式生成
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    new_tokens = outputs[0][prompt_tokens:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    completion_tokens = len(new_tokens)

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[
            ChatChoice(
                message=Message(role="assistant", content=response_text)
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def stream_generate(
    gen_kwargs: dict, prompt_tokens: int, request: ChatRequest
) -> AsyncGenerator[str, None]:
    """流式生成"""
    import asyncio
    import json

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # 使用 TextIteratorStreamer
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs["streamer"] = streamer

    # 在后台线程中运行生成
    thread = Thread(target=lambda: model.generate(**gen_kwargs))
    thread.start()

    # 发送角色信息
    chunk = StreamResponse(
        id=chat_id,
        created=created,
        model=MODEL_NAME,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"

    # 流式输出 token
    completion_tokens = 0
    for text_chunk in streamer:
        if text_chunk:
            completion_tokens += 1
            chunk = StreamResponse(
                id=chat_id,
                created=created,
                model=MODEL_NAME,
                choices=[StreamChoice(delta=DeltaMessage(content=text_chunk))],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            await asyncio.sleep(0)  # 让出控制权

    # 发送结束标记
    chunk = StreamResponse(
        id=chat_id,
        created=created,
        model=MODEL_NAME,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

    thread.join()


# ---- 主函数 ----

def main():
    global model, tokenizer, MODEL_NAME

    parser = argparse.ArgumentParser(description="Qwen3.6 Transformers API Server")
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/outputs/qwen3.6-27b-merged",
        help="合并后模型路径",
    )
    parser.add_argument("--port", type=int, default=6006, help="服务端口")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument(
        "--model-name", default="qwen3.6-27b-nsfw", help="对外展示的模型名称"
    )
    parser.add_argument(
        "--max-length", type=int, default=8192, help="最大上下文长度（仅用于提示）"
    )
    args = parser.parse_args()

    MODEL_NAME = args.model_name

    print("=" * 50)
    print("  Qwen3.6 Transformers API Server")
    print("=" * 50)
    print(f"  模型路径: {args.model_path}")
    print(f"  模型名称: {MODEL_NAME}")
    print(f"  端口: {args.port}")
    print(f"  最大长度: {args.max_length}")
    print("")

    print("[1/2] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    print("[2/2] 加载模型（bfloat16, device_map=auto）...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("")
    print("=" * 50)
    print(f"  模型加载完成!")
    print(f"  API 地址: http://{args.host}:{args.port}/v1")
    print(f"  健康检查: http://{args.host}:{args.port}/health")
    print("=" * 50)
    print("")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
