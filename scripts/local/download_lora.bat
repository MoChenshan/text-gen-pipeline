@echo off
REM ============================================================
REM 从远端下载 LoRA 权重到本地
REM 用途: 训练完成后，将 LoRA 权重下载到本地备份
REM 说明: 只下载必要文件（不含 checkpoint 目录，节省空间）
REM 使用: download_lora.bat
REM ============================================================

echo ==========================================
echo   从 AutoDL 下载 LoRA 权重
echo ==========================================

REM ---- 配置 ----
SET REMOTE_HOST=root@connect.westd.seetacloud.com
SET REMOTE_PORT=42655
SET REMOTE_LORA_DIR=/root/autodl-tmp/outputs/qwen3.6-27b-lora
SET LOCAL_OUTPUT_DIR=D:\Project\Python\text-gen-pipeline\outputs\lora

REM 创建本地目录
if not exist "%LOCAL_OUTPUT_DIR%" mkdir "%LOCAL_OUTPUT_DIR%"

echo.
echo 下载 LoRA 权重（仅必要文件，不含 checkpoint）...
echo   远端: %REMOTE_HOST%:%REMOTE_LORA_DIR%
echo   本地: %LOCAL_OUTPUT_DIR%
echo.

REM 下载核心权重文件
echo [1/6] adapter_config.json
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% %REMOTE_HOST%:%REMOTE_LORA_DIR%/adapter_config.json "%LOCAL_OUTPUT_DIR%\"

echo [2/6] adapter_model.safetensors（约 1-2 GB，请耐心等待）
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% %REMOTE_HOST%:%REMOTE_LORA_DIR%/adapter_model.safetensors "%LOCAL_OUTPUT_DIR%\"

REM 下载 tokenizer 相关文件
echo [3/6] tokenizer.json
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% %REMOTE_HOST%:%REMOTE_LORA_DIR%/tokenizer.json "%LOCAL_OUTPUT_DIR%\"

echo [4/6] tokenizer_config.json
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% %REMOTE_HOST%:%REMOTE_LORA_DIR%/tokenizer_config.json "%LOCAL_OUTPUT_DIR%\"

echo [5/6] chat_template.jinja
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% %REMOTE_HOST%:%REMOTE_LORA_DIR%/chat_template.jinja "%LOCAL_OUTPUT_DIR%\"

REM 下载训练记录（可选，方便查看）
echo [6/6] 训练记录文件
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% %REMOTE_HOST%:%REMOTE_LORA_DIR%/training_loss.png "%LOCAL_OUTPUT_DIR%\" 2>nul
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% %REMOTE_HOST%:%REMOTE_LORA_DIR%/trainer_state.json "%LOCAL_OUTPUT_DIR%\" 2>nul
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% %REMOTE_HOST%:%REMOTE_LORA_DIR%/trainer_log.jsonl "%LOCAL_OUTPUT_DIR%\" 2>nul

echo.
echo ==========================================
echo   下载完成!
echo ==========================================
echo   LoRA 权重保存在: %LOCAL_OUTPUT_DIR%
echo   （已跳过 checkpoint 目录，节省数 GB 空间）
echo.
pause
