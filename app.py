#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI工业缺陷检测系统 - GUI启动脚本
自动处理环境变量和线程问题
"""

import os
import sys

# 设置环境变量以解决 OpenMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 禁用 MKL 以避免线程冲突
os.environ['MKL_SERVICE_FORCE_INTEL'] = '0'

from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from gui.main_window import MainWindow


def main():
    """主函数"""
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("AI工业缺陷检测系统")
    app.setOrganizationName("AITool")
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
