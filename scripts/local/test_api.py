"""
API 连通性测试脚本
用于测试远端 vLLM 服务是否正常工作

使用方法:
    python test_api.py --api_url http://your-autodl-url:6006/v1
"""

import argparse
import json
import requests


def test_models(api_url: str, api_key: str = "EMPTY"):
    """测试模型列表接口"""
    print("[测试] 获取模型列表...")
    try:
        resp = requests.get(
            f"{api_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.status_code == 200:
            models = resp.json()
            print(f"  ✓ 成功! 可用模型: {[m['id'] for m in models['data']]}")
            return True
        else:
            print(f"  ✗ 失败! 状态码: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return False


def test_chat_completion(api_url: str, api_key: str = "EMPTY"):
    """测试对话补全接口"""
    print("\n[测试] 对话补全...")
    try:
        resp = requests.post(
            f"{api_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen3-14b-nsfw",
                "messages": [
                    {"role": "user", "content": "你好，请简单介绍一下你自己。"}
                ],
                "max_tokens": 200,
                "temperature": 0.7,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            print(f"  ✓ 成功!")
            print(f"  回复: {content[:200]}...")
            return True
        else:
            print(f"  ✗ 失败! 状态码: {resp.status_code}")
            print(f"  响应: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
        return False


def test_completion(api_url: str, api_key: str = "EMPTY"):
    """测试文本补全/续写接口"""
    print("\n[测试] 文本续写...")
    try:
        resp = requests.post(
            f"{api_url}/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen3-14b-nsfw",
                "prompt": "夜色渐深，月光透过窗帘洒在地板上，她轻轻推开了房门",
                "max_tokens": 300,
                "temperature": 0.8,
                "top_p": 0.9,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            result = resp.json()
            text = result["choices"][0]["text"]
            print(f"  ✓ 成功!")
            print(f"  续写: {text[:300]}...")
            return True
        else:
            print(f"  ✗ 失败! 状态码: {resp.status_code}")
            print(f"  响应: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ 请求失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="API 连通性测试")
    parser.add_argument("--api_url", type=str, required=True,
                        help="API 地址，如 http://your-url:6006/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API Key (默认: EMPTY)")
    args = parser.parse_args()
    
    print("==========================================")
    print("  vLLM API 连通性测试")
    print("==========================================")
    print(f"  API 地址: {args.api_url}")
    print("")
    
    results = []
    results.append(("模型列表", test_models(args.api_url, args.api_key)))
    results.append(("对话补全", test_chat_completion(args.api_url, args.api_key)))
    results.append(("文本续写", test_completion(args.api_url, args.api_key)))
    
    print("\n==========================================")
    print("  测试结果汇总")
    print("==========================================")
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n  🎉 所有测试通过! API 服务正常工作。")
        print("\n  下一步: 配置 SillyTavern 连接此 API")
        print(f"    API URL: {args.api_url}")
        print(f"    API Key: {args.api_key}")
    else:
        print("\n  ⚠️ 部分测试失败，请检查服务状态。")


if __name__ == "__main__":
    main()
