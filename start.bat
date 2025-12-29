@echo off
REM AI工业缺陷检测系统 - GUI启动脚本
REM 使用方法: 双击运行此文件

echo ========================================
echo AI工业缺陷检测系统 - GUI
echo ========================================
echo.

REM 激活conda环境
echo 正在激活 defect_detection 环境...
call conda activate defect_detection

REM 启动应用
echo 启动GUI应用...
echo.
python app.py

pause
