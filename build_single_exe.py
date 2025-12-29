#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 PyInstaller 打包 app.py 为单文件 EXE
包含所有依赖和数据文件
"""

import os
import sys
import subprocess
from pathlib import Path

def build_exe():
    """构建单文件EXE"""
    
    # 获取项目路径
    project_root = Path(__file__).parent
    app_py = project_root / 'app.py'
    output_dir = project_root / 'dist'
    
    # PyInstaller 命令参数 - 优化版本，减少构建时间和文件大小
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',  # 单文件模式
        '--windowed',  # 不显示控制台窗口
        '--name', 'AITool',  # EXE 名称
        '--distpath', str(output_dir),  # 输出目录
        '--workpath', str(project_root / 'build_temp'),  # 临时构建目录
        '--specpath', str(project_root),  # spec 文件位置
        '--clean',  # 清理临时文件
        
        # 添加数据文件
        '--add-data', f'{project_root / "config.yaml"}{os.pathsep}.',
        '--add-data', f'{project_root / "data.yaml"}{os.pathsep}.',
        
        # 隐藏导入 - 只导入必要的
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtGui',
        '--hidden-import=PySide6.QtWidgets',
        '--hidden-import=ultralytics',
        '--hidden-import=torch',
        '--hidden-import=cv2',
        '--hidden-import=yaml',
        '--hidden-import=numpy',
        '--hidden-import=onnx',
        '--hidden-import=onnxruntime',
        '--hidden-import=PIL',
        '--hidden-import=tqdm',
        
        # 优化选项
        '--noupx',  # 禁用 UPX (避免压缩问题)
        '-y',  # 自动覆盖
        
        # 收集数据
        '--collect-all=PySide6',
        '--collect-all=ultralytics',
        '--collect-all=cv2',
        '--collect-all=yaml',
        
        # 入口点
        str(app_py),
    ]
    
    print(f"开始构建单文件 EXE...")
    print(f"项目路径: {project_root}")
    print(f"输出目录: {output_dir}")
    print()
    
    try:
        # 运行 PyInstaller
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            exe_path = output_dir / 'AITool.exe'
            print(f"\n✓ 构建成功!")
            print(f"EXE 文件位置: {exe_path}")
            print(f"文件大小: {exe_path.stat().st_size / (1024**2):.1f} MB")
        else:
            print(f"\n✗ 构建失败，返回码: {result.returncode}")
            sys.exit(1)
            
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    build_exe()
