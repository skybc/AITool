"""
图片查看器 - ImageViewer
支持图片缩放、拖动、检测结果绘制
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QLabel
from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QWheelEvent, QMouseEvent


class ImageViewer(QWidget):
    """图片查看器组件"""
    
    def __init__(self):
        super().__init__()
        self.image = None  # 原始图像 (numpy array)
        self.detections = []  # 检测结果列表
        self.scale = 1.0  # 缩放比例
        self.offset = QPoint(0, 0)  # 偏移量
        self.last_pos = QPoint()  # 鼠标上次位置
        self.is_dragging = False  # 是否正在拖动
        
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
    
    def load_image(self, image_path):
        """加载图片
        
        Args:
            image_path: 图片路径
        """
        try:
            # 使用opencv读取（支持中文路径）
            self.image = cv2.imdecode(
                np.fromfile(image_path, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if self.image is None:
                return False
            
            # 重置视图
            self.scale = 1.0
            self.offset = QPoint(0, 0)
            self.detections = []
            
            # 自适应缩放
            self._fit_to_window()
            self.update()
            return True
            
        except Exception as e:
            print(f"加载图片失败: {e}")
            return False
    
    def set_detections(self, detections):
        """设置检测结果
        
        Args:
            detections: 检测结果列表，每个元素包含 bbox, class_name, confidence
        """
        self.detections = detections
        self.update()
    
    def clear(self):
        """清空图片"""
        self.image = None
        self.detections = []
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.update()
    
    def _fit_to_window(self):
        """自适应窗口大小"""
        if self.image is None:
            return
        
        img_h, img_w = self.image.shape[:2]
        widget_w = self.width()
        widget_h = self.height()
        
        # 计算缩放比例（留10像素边距）
        scale_w = (widget_w - 20) / img_w
        scale_h = (widget_h - 20) / img_h
        self.scale = min(scale_w, scale_h, 1.0)  # 不放大，只缩小
        
        # 居中显示
        scaled_w = int(img_w * self.scale)
        scaled_h = int(img_h * self.scale)
        self.offset = QPoint(
            (widget_w - scaled_w) // 2,
            (widget_h - scaled_h) // 2
        )
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 填充背景
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        if self.image is None:
            # 显示提示信息
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "请从左侧选择图片"
            )
            return
        
        # 绘制图片
        img_h, img_w = self.image.shape[:2]
        scaled_w = int(img_w * self.scale)
        scaled_h = int(img_h * self.scale)
        
        # 转换为QImage
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # 如果有检测结果，先在numpy数组上绘制
        if self.detections:
            rgb_image = self._draw_detections_on_image(rgb_image)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 缩放并绘制
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            scaled_w,
            scaled_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        painter.drawPixmap(self.offset, scaled_pixmap)
    
    def _draw_detections_on_image(self, rgb_image):
        """在图片上绘制检测结果
        
        Args:
            rgb_image: RGB格式图像 (numpy array)
            
        Returns:
            绘制后的图像
        """
        img = rgb_image.copy()
        
        for det in self.detections:
            x_min, y_min, x_max, y_max = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # 绘制边框（绿色）
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name} {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # 获取文本尺寸
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # 绘制文本背景
            cv2.rectangle(
                img,
                (x_min, y_min - text_h - 10),
                (x_min + text_w, y_min),
                (0, 255, 0),
                -1
            )
            
            # 绘制文本
            cv2.putText(
                img,
                label,
                (x_min, y_min - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )
        
        return img
    
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮事件 - 缩放"""
        if self.image is None:
            return
        
        # 获取鼠标位置
        mouse_pos = event.position().toPoint()
        
        # 计算缩放前鼠标在图片上的相对位置
        old_pos = (mouse_pos - self.offset) / self.scale
        
        # 缩放
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale *= 1.1  # 放大
        else:
            self.scale *= 0.9  # 缩小
        
        # 限制缩放范围
        self.scale = max(0.1, min(self.scale, 10.0))
        
        # 计算新的偏移量，使鼠标位置保持不变
        new_offset = mouse_pos - old_pos * self.scale
        self.offset = new_offset.toPoint()
        
        self.update()
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.RightButton:
            self.is_dragging = True
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件 - 拖动"""
        if self.is_dragging:
            delta = event.pos() - self.last_pos
            self.offset += delta
            self.last_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.RightButton:
            self.is_dragging = False
            self.setCursor(Qt.ArrowCursor)
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        if self.image is not None and self.scale == 1.0:
            self._fit_to_window()
        super().resizeEvent(event)
