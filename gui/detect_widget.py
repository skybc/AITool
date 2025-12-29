"""
检测界面 - DetectWidget
包含文件选择、模型选择、检测功能和结果显示
"""

import os
from pathlib import Path
from typing import Dict, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QGroupBox,
    QCheckBox, QMessageBox, QSplitter, QProgressDialog
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.infer import DefectInference, DetectionResult
from .image_viewer import ImageViewer


class InferenceThread(QThread):
    """推理线程"""
    progress_signal = Signal(int, int)  # current, total
    result_signal = Signal(str, list)  # image_path, detections
    finished_signal = Signal(bool, str)  # success, message
    
    def __init__(self, inferencer, image_paths):
        super().__init__()
        self.inferencer = inferencer
        self.image_paths = image_paths
        self.is_running = True
    
    def run(self):
        """执行推理"""
        try:
            import cv2
            total = len(self.image_paths)
            
            successful = 0
            failed = 0
            
            for idx, img_path in enumerate(self.image_paths):
                if not self.is_running:
                    self.finished_signal.emit(False, "用户取消")
                    return
                
                try:
                    # 读取图片
                    image = cv2.imread(img_path)
                    if image is None:
                        failed += 1
                        self.progress_signal.emit(idx + 1, total)
                        continue
                    
                    # 推理
                    detections = self.inferencer.infer(image)
                    
                    # 转换为字典格式
                    det_dicts = []
                    for det in detections:
                        det_dicts.append({
                            'bbox': det.bbox,
                            'class_name': det.class_name,
                            'confidence': det.confidence,
                            'class_id': det.class_id
                        })
                    
                    # 发送结果
                    self.result_signal.emit(img_path, det_dicts)
                    successful += 1
                    self.progress_signal.emit(idx + 1, total)
                    
                except Exception as e:
                    print(f"推理图片失败 {img_path}: {e}")
                    failed += 1
                    self.progress_signal.emit(idx + 1, total)
            
            message = f"检测完成，成功处理 {successful} 张图片"
            if failed > 0:
                message += f"，失败 {failed} 张"
            
            self.finished_signal.emit(True, message)
            
        except Exception as e:
            self.finished_signal.emit(False, f"推理出错: {str(e)}")
    
    def stop(self):
        """停止推理"""
        self.is_running = False


class DetectWidget(QWidget):
    """检测界面组件"""
    
    # 支持的图片格式
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.inferencer = None
        self.image_results = {}  # {图片路径: [检测结果]}
        self.current_image_path = None
        self.inference_thread = None
        
        self._init_ui()
        self._setup_connections()
    
    def _init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # ============== 第一行：按钮 ==============
        button_layout = QHBoxLayout()
        
        self.add_files_btn = QPushButton("选择文件")
        self.add_files_btn.setMinimumHeight(40)
        self.add_files_btn.setToolTip("添加图片文件到列表")
        button_layout.addWidget(self.add_files_btn)
        
        self.remove_files_btn = QPushButton("移除文件")
        self.remove_files_btn.setMinimumHeight(40)
        self.remove_files_btn.setToolTip("从列表中移除选中的文件")
        button_layout.addWidget(self.remove_files_btn)
        
        self.add_folder_btn = QPushButton("选择目录")
        self.add_folder_btn.setMinimumHeight(40)
        self.add_folder_btn.setToolTip("添加目录中的所有图片")
        button_layout.addWidget(self.add_folder_btn)
        
        self.select_model_btn = QPushButton("选择模型")
        self.select_model_btn.setMinimumHeight(40)
        self.select_model_btn.setToolTip("选择训练好的模型文件 (.pt)")
        button_layout.addWidget(self.select_model_btn)
        
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.setMinimumHeight(40)
        self.detect_btn.setEnabled(False)
        self.detect_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.detect_btn)
        
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        # ============== 第二行：信息显示 ==============
        info_layout = QHBoxLayout()
        
        # 模型信息
        info_layout.addWidget(QLabel("模型:"))
        self.model_label = QLabel("未选择")
        self.model_label.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(self.model_label, 1)
        
        info_layout.addSpacing(20)
        
        # 模型类型
        info_layout.addWidget(QLabel("类型:"))
        self.model_type_label = QLabel("-")
        info_layout.addWidget(self.model_type_label)
        
        info_layout.addSpacing(20)
        
        # 全部检测复选框
        self.detect_all_checkbox = QCheckBox("检测全部图片")
        self.detect_all_checkbox.setChecked(True)
        self.detect_all_checkbox.setToolTip("勾选：检测所有图片；不勾选：只检测选中的图片")
        info_layout.addWidget(self.detect_all_checkbox)
        
        info_layout.addStretch()
        
        main_layout.addLayout(info_layout)
        
        # ============== 第三行：文件树 + 图片显示（占满剩余空间） ==============
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：文件树
        file_tree_group = QGroupBox("文件列表")
        file_tree_layout = QVBoxLayout()
        
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["文件名", "状态"])
        self.file_tree.setColumnWidth(0, 250)
        self.file_tree.setAlternatingRowColors(True)
        file_tree_layout.addWidget(self.file_tree)
        
        # 文件统计
        self.file_count_label = QLabel("文件数: 0")
        self.file_count_label.setStyleSheet("color: #666; padding: 5px;")
        file_tree_layout.addWidget(self.file_count_label)
        
        file_tree_group.setLayout(file_tree_layout)
        splitter.addWidget(file_tree_group)
        
        # 右侧：图片查看器
        viewer_group = QGroupBox("图片预览")
        viewer_layout = QVBoxLayout()
        
        self.image_viewer = ImageViewer()
        viewer_layout.addWidget(self.image_viewer)
        
        # 图片信息
        self.image_info_label = QLabel("鼠标滚轮缩放 | 右键拖动移动")
        self.image_info_label.setStyleSheet("color: #666; padding: 5px;")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        viewer_layout.addWidget(self.image_info_label)
        
        viewer_group.setLayout(viewer_layout)
        splitter.addWidget(viewer_group)
        
        # 设置分割比例
        splitter.setStretchFactor(0, 2)  # 文件树占2份
        splitter.setStretchFactor(1, 5)  # 图片显示占5份
        
        main_layout.addWidget(splitter, 1)  # 占满剩余空间
    
    def _setup_connections(self):
        """设置信号连接"""
        self.add_files_btn.clicked.connect(self.add_files)
        self.remove_files_btn.clicked.connect(self.remove_files)
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.select_model_btn.clicked.connect(self.select_model)
        self.detect_btn.clicked.connect(self.start_detection)
        self.file_tree.itemClicked.connect(self.on_file_selected)
    
    def add_files(self):
        """添加图片文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片文件",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;所有文件 (*.*)"
        )
        
        if files:
            added_count = 0
            for file_path in files:
                if self._add_file_to_tree(file_path):
                    added_count += 1
            
            self._update_file_count()
            if added_count > 0:
                QMessageBox.information(self, "添加成功", f"已添加 {added_count} 个文件")
    
    def add_folder(self):
        """添加目录中的所有图片"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择图片目录",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            # 递归查找所有图片文件
            image_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if Path(file).suffix.lower() in self.SUPPORTED_FORMATS:
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                QMessageBox.warning(self, "警告", "所选目录中没有找到支持的图片文件")
                return
            
            # 添加到树中
            added_count = 0
            for file_path in image_files:
                if self._add_file_to_tree(file_path, directory):
                    added_count += 1
            
            self._update_file_count()
            if added_count > 0:
                QMessageBox.information(self, "添加成功", f"已添加 {added_count} 个文件")
    
    def _add_file_to_tree(self, file_path, root_dir=None):
        """添加文件到树中
        
        Args:
            file_path: 文件路径
            root_dir: 根目录（用于创建树结构）
            
        Returns:
            是否添加成功
        """
        # 检查是否已存在
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            if self._get_item_path(item) == file_path:
                return False  # 已存在
        
        # 创建树节点
        if root_dir:
            # 如果指定了根目录，创建目录结构
            rel_path = os.path.relpath(file_path, root_dir)
            parts = Path(rel_path).parts
            
            # 查找或创建父节点
            parent = None
            current_path = root_dir
            
            for part in parts[:-1]:  # 除了文件名的所有部分
                current_path = os.path.join(current_path, part)
                parent = self._find_or_create_folder_item(part, parent)
            
            # 添加文件节点
            file_item = QTreeWidgetItem(parent if parent else self.file_tree)
            file_item.setText(0, parts[-1])
            file_item.setText(1, "未检测")
            file_item.setData(0, Qt.UserRole, file_path)
        else:
            # 直接添加文件
            file_item = QTreeWidgetItem(self.file_tree)
            file_item.setText(0, os.path.basename(file_path))
            file_item.setText(1, "未检测")
            file_item.setData(0, Qt.UserRole, file_path)
        
        return True
    
    def _find_or_create_folder_item(self, folder_name, parent):
        """查找或创建文件夹节点"""
        # 在父节点下查找
        if parent:
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.text(0) == folder_name and child.childCount() >= 0:
                    return child
            # 未找到，创建新节点
            folder_item = QTreeWidgetItem(parent)
        else:
            # 在顶层查找
            for i in range(self.file_tree.topLevelItemCount()):
                item = self.file_tree.topLevelItem(i)
                if item.text(0) == folder_name and item.childCount() >= 0:
                    return item
            # 未找到，创建新节点
            folder_item = QTreeWidgetItem(self.file_tree)
        
        folder_item.setText(0, folder_name)
        folder_item.setText(1, "-")
        return folder_item
    
    def _get_item_path(self, item):
        """获取树节点对应的文件路径"""
        return item.data(0, Qt.UserRole)
    
    def _is_file_item(self, item):
        """判断是否是文件节点"""
        return item.data(0, Qt.UserRole) is not None
    
    def remove_files(self):
        """移除选中的文件"""
        selected_items = self.file_tree.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要移除的文件")
            return
        
        # 移除选中的项
        for item in selected_items:
            if self._is_file_item(item):
                # 从结果字典中移除
                file_path = self._get_item_path(item)
                if file_path in self.image_results:
                    del self.image_results[file_path]
                
                # 从树中移除
                parent = item.parent()
                if parent:
                    parent.removeChild(item)
                else:
                    index = self.file_tree.indexOfTopLevelItem(item)
                    self.file_tree.takeTopLevelItem(index)
        
        self._update_file_count()
        self.image_viewer.clear()
        self.current_image_path = None
    
    def _update_file_count(self):
        """更新文件计数"""
        count = self._count_files()
        self.file_count_label.setText(f"文件数: {count}")
    
    def _count_files(self):
        """统计文件数量"""
        count = 0
        
        def count_recursive(parent):
            nonlocal count
            child_count = parent.childCount() if hasattr(parent, 'childCount') else parent.topLevelItemCount()
            for i in range(child_count):
                item = parent.child(i) if hasattr(parent, 'child') else parent.topLevelItem(i)
                if self._is_file_item(item):
                    count += 1
                else:
                    count_recursive(item)
        
        count_recursive(self.file_tree)
        return count
    
    def _get_all_file_paths(self):
        """获取所有文件路径"""
        file_paths = []
        
        def collect_recursive(parent):
            child_count = parent.childCount() if hasattr(parent, 'childCount') else parent.topLevelItemCount()
            for i in range(child_count):
                item = parent.child(i) if hasattr(parent, 'child') else parent.topLevelItem(i)
                if self._is_file_item(item):
                    file_paths.append(self._get_item_path(item))
                else:
                    collect_recursive(item)
        
        collect_recursive(self.file_tree)
        return file_paths
    
    def select_model(self):
        """选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            "模型文件 (*.pt *.onnx);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                # 判断是否是ONNX模型
                use_onnx = file_path.endswith('.onnx')
                
                # 创建推理器
                self.inferencer = DefectInference(
                    weights=file_path,
                    conf_threshold=0.25,
                    iou_threshold=0.45,
                    device='cuda',
                    use_onnx=use_onnx
                )
                
                self.model_path = file_path
                self.model_label.setText(os.path.basename(file_path))
                self.model_label.setStyleSheet("color: #000;")
                
                # 显示模型类型
                model_type = "ONNX" if use_onnx else "PyTorch"
                self.model_type_label.setText(model_type)
                
                # 启用检测按钮
                self.detect_btn.setEnabled(True)
                
                QMessageBox.information(self, "模型加载成功", f"已加载 {model_type} 模型")
                
            except Exception as e:
                QMessageBox.critical(self, "模型加载失败", f"无法加载模型:\n{str(e)}")
    
    def on_file_selected(self, item, column):
        """文件被选中时的回调"""
        if not self._is_file_item(item):
            return
        
        file_path = self._get_item_path(item)
        self.current_image_path = file_path
        
        # 加载图片
        if self.image_viewer.load_image(file_path):
            # 如果有检测结果，显示结果
            if file_path in self.image_results:
                self.image_viewer.set_detections(self.image_results[file_path])
                
                # 更新信息标签
                det_count = len(self.image_results[file_path])
                self.image_info_label.setText(
                    f"检测到 {det_count} 个目标 | 鼠标滚轮缩放 | 右键拖动移动"
                )
            else:
                self.image_info_label.setText("鼠标滚轮缩放 | 右键拖动移动")
    
    def start_detection(self):
        """开始检测"""
        if not self.inferencer:
            QMessageBox.warning(self, "错误", "请先选择模型！")
            return
        
        # 获取要检测的图片列表
        if self.detect_all_checkbox.isChecked():
            # 检测全部
            image_paths = self._get_all_file_paths()
        else:
            # 只检测选中的
            selected_items = self.file_tree.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "错误", "请先选择要检测的文件！")
                return
            
            image_paths = []
            for item in selected_items:
                if self._is_file_item(item):
                    image_paths.append(self._get_item_path(item))
        
        if not image_paths:
            QMessageBox.warning(self, "错误", "没有可检测的文件！")
            return
        
        # 创建进度对话框
        self.progress_dialog = QProgressDialog(
            "正在检测...",
            "取消",
            0,
            len(image_paths),
            self
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setWindowTitle("检测进度")
        self.progress_dialog.canceled.connect(self.stop_detection)
        
        # 创建并启动推理线程
        self.inference_thread = InferenceThread(self.inferencer, image_paths)
        self.inference_thread.progress_signal.connect(self.on_detection_progress)
        self.inference_thread.result_signal.connect(self.on_detection_result)
        self.inference_thread.finished_signal.connect(self.on_detection_finished)
        
        # 禁用按钮
        self.detect_btn.setEnabled(False)
        self.select_model_btn.setEnabled(False)
        self.add_files_btn.setEnabled(False)
        self.add_folder_btn.setEnabled(False)
        
        self.inference_thread.start()
    
    def stop_detection(self):
        """停止检测"""
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.stop()
            self.inference_thread.wait()
    
    def on_detection_progress(self, current, total):
        """检测进度更新"""
        self.progress_dialog.setValue(current)
    
    def on_detection_result(self, image_path, detections):
        """单张图片检测完成"""
        # 保存结果
        self.image_results[image_path] = detections
        
        # 更新树中的状态
        def update_tree_status(parent):
            child_count = parent.childCount() if hasattr(parent, 'childCount') else parent.topLevelItemCount()
            for i in range(child_count):
                item = parent.child(i) if hasattr(parent, 'child') else parent.topLevelItem(i)
                if self._is_file_item(item) and self._get_item_path(item) == image_path:
                    det_count = len(detections)
                    item.setText(1, f"检测到 {det_count} 个目标")
                    return True
                elif item.childCount() > 0:
                    if update_tree_status(item):
                        return True
            return False
        
        update_tree_status(self.file_tree)
        
        # 如果是当前显示的图片，更新显示
        if image_path == self.current_image_path:
            self.image_viewer.set_detections(detections)
            det_count = len(detections)
            self.image_info_label.setText(
                f"检测到 {det_count} 个目标 | 鼠标滚轮缩放 | 右键拖动移动"
            )
    
    def on_detection_finished(self, success, message):
        """检测完成"""
        self.progress_dialog.close()
        
        # 恢复按钮
        self.detect_btn.setEnabled(True)
        self.select_model_btn.setEnabled(True)
        self.add_files_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "检测完成", message)
        else:
            QMessageBox.warning(self, "检测失败", message)
