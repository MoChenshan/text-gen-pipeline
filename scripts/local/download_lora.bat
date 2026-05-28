@echo off
REM ============================================================
REM 从远端下载 LoRA 权重到本地
REM 用途: 训练完成后，将 LoRA 权重下载到本地备份
REM 使用: download_lora.bat
REM ============================================================

echo ==========================================
echo   从 AutoDL 下载 LoRA 权重
echo ==========================================

REM ---- 配置 ----
SET REMOTE_HOST=root@connect.westb.seetacloud.com
SET REMOTE_PORT=xxxxx
SET REMOTE_LORA_DIR=/root/autodl-tmp/outputs/qwen3.6-27b-lora
SET LOCAL_OUTPUT_DIR=D:\Project\Python\text-gen-pipeline\outputs\lora

REM 创建本地目录
if not exist "%LOCAL_OUTPUT_DIR%" mkdir "%LOCAL_OUTPUT_DIR%"

echo.
echo 下载 LoRA 权重...
echo   远端: %REMOTE_HOST%:%REMOTE_LORA_DIR%
echo   本地: %LOCAL_OUTPUT_DIR%
echo.

"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% -r %REMOTE_HOST%:%REMOTE_LORA_DIR%/* "%LOCAL_OUTPUT_DIR%\"

echo.
echo ==========================================
echo   下载完成!
echo ==========================================
echo   LoRA 权重保存在: %LOCAL_OUTPUT_DIR%
echo.
pause
