"""
主窗口 - MainWindow
包含菜单栏、工具栏和工作区切换
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QMenuBar, QStackedWidget, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon

from .train_widget import TrainWidget
from .detect_widget import DetectWidget


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI工业缺陷检测系统 - YOLO11")
        self.setMinimumSize(1200, 800)
        
        # 初始化UI
        self._init_ui()
        self._setup_connections()
        
        # 默认显示训练界面
        self.show_train_widget()
    
    def _init_ui(self):
        """初始化UI"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建菜单栏（隐藏）
        self._create_menu_bar()
        
        # 创建工具栏
        self._create_toolbar()
        
        # 创建堆叠窗口部件（用于切换界面）
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # 创建训练界面
        self.train_widget = TrainWidget()
        self.stacked_widget.addWidget(self.train_widget)
        
        # 创建检测界面
        self.detect_widget = DetectWidget()
        self.stacked_widget.addWidget(self.detect_widget)
    
    def _create_menu_bar(self):
        """创建菜单栏（预留，先隐藏）"""
        menubar = self.menuBar()
        menubar.setVisible(False)  # 隐藏菜单栏
        
        # 预留菜单项
        file_menu = menubar.addMenu("文件(&F)")
        edit_menu = menubar.addMenu("编辑(&E)")
        view_menu = menubar.addMenu("视图(&V)")
        help_menu = menubar.addMenu("帮助(&H)")
    
    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        
        # 训练按钮
        self.train_action = QAction("训练", self)
        self.train_action.setToolTip("切换到训练界面")
        self.train_action.setCheckable(True)
        self.train_action.setChecked(True)
        toolbar.addAction(self.train_action)
        
        toolbar.addSeparator()
        
        # 检测按钮
        self.detect_action = QAction("检测", self)
        self.detect_action.setToolTip("切换到检测界面")
        self.detect_action.setCheckable(True)
        toolbar.addAction(self.detect_action)
    
    def _setup_connections(self):
        """设置信号连接"""
        self.train_action.triggered.connect(self.show_train_widget)
        self.detect_action.triggered.connect(self.show_detect_widget)
    
    def show_train_widget(self):
        """显示训练界面"""
        self.stacked_widget.setCurrentWidget(self.train_widget)
        self.train_action.setChecked(True)
        self.detect_action.setChecked(False)
    
    def show_detect_widget(self):
        """显示检测界面"""
        self.stacked_widget.setCurrentWidget(self.detect_widget)
        self.train_action.setChecked(False)
        self.detect_action.setChecked(True)
    
    def closeEvent(self, event):
        """关闭事件"""
        # 如果正在训练或检测，提示用户
        if hasattr(self.train_widget, 'is_training') and self.train_widget.is_training:
            reply = QMessageBox.question(
                self,
                '确认退出',
                '训练正在进行中，确定要退出吗？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        event.accept()
