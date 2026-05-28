@echo off
REM ============================================================
REM 本地同步脚本 - 将本地文件同步到 AutoDL 远端
REM 用途: 同步代码和专有数据集到远端训练机器
REM 使用: sync_to_remote.bat
REM 
REM 前置要求:
REM   1. 安装 scp/rsync (通过 Git Bash 或 WSL)
REM   2. 配置 SSH 密钥免密登录
REM ============================================================

echo ==========================================
echo   同步本地文件到 AutoDL 远端
echo ==========================================

REM ---- 配置 ----
REM 请修改为你的 AutoDL SSH 地址
SET REMOTE_HOST=root@connect.westd.seetacloud.com
SET REMOTE_PORT=42655
SET REMOTE_PROJECT_DIR=/root/project/text-gen-pipeline
SET REMOTE_DATASET_DIR=/root/autodl-tmp/datasets/proprietary

REM 本地路径
SET LOCAL_PROJECT_DIR=D:\Project\Python\text-gen-pipeline
SET LOCAL_DATA_DIR=%LOCAL_PROJECT_DIR%\data\processed

echo.
echo [1/3] 同步项目代码...
echo   本地: %LOCAL_PROJECT_DIR%\scripts
echo   远端: %REMOTE_HOST%:%REMOTE_PROJECT_DIR%/scripts
echo.

REM 使用 scp 同步代码（通过 Git Bash 的 scp）
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% -r "%LOCAL_PROJECT_DIR%\scripts" %REMOTE_HOST%:%REMOTE_PROJECT_DIR%/
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% -r "%LOCAL_PROJECT_DIR%\configs" %REMOTE_HOST%:%REMOTE_PROJECT_DIR%/

echo.
echo [2/3] 同步专有数据集...
echo   本地: %LOCAL_DATA_DIR%
echo   远端: %REMOTE_HOST%:%REMOTE_DATASET_DIR%
echo.

if exist "%LOCAL_DATA_DIR%" (
    "C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% -r "%LOCAL_DATA_DIR%\*" %REMOTE_HOST%:%REMOTE_DATASET_DIR%/
    echo   完成!
) else (
    echo   警告: 本地数据目录不存在: %LOCAL_DATA_DIR%
    echo   请先准备专有数据集
)

echo.
echo [3/3] 同步配置文件...
"C:\Program Files\Git\usr\bin\scp.exe" -P %REMOTE_PORT% -r "%LOCAL_PROJECT_DIR%\configs\*" %REMOTE_HOST%:%REMOTE_PROJECT_DIR%/configs/

echo.
echo ==========================================
echo   同步完成!
echo ==========================================
echo.
echo 下一步:
echo   1. SSH 登录远端: ssh -p %REMOTE_PORT% %REMOTE_HOST%
echo   2. 运行训练: bash %REMOTE_PROJECT_DIR%/scripts/remote/run_train.sh
echo.
pause
