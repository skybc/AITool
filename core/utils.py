"""
工具函数库
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import cv2


def load_config(config_path: str) -> Dict:
    """加载 YAML 配置文件"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(data: Dict, path: str):
    """保存 JSON 文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict:
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """获取目录下的所有图像文件"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    img_dir = Path(directory)
    image_files = []
    
    for ext in extensions:
        image_files.extend(sorted(img_dir.glob(f'*{ext}')))
        image_files.extend(sorted(img_dir.glob(f'*{ext.upper()}')))
    
    return [str(f) for f in sorted(set(image_files))]


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """计算两个边界框的 IoU
    
    参数:
        box1, box2: (x_min, y_min, x_max, y_max)
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    
    # 交集
    inter_xmin = max(x_min1, x_min2)
    inter_ymin = max(y_min1, y_min2)
    inter_xmax = min(x_max1, x_max2)
    inter_ymax = min(y_max1, y_max2)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # 并集
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def non_max_suppression(
    detections: List[Dict],
    iou_threshold: float = 0.45
) -> List[Dict]:
    """非极大值抑制 (NMS)
    
    参数:
        detections: 检测结果列表，每个元素包含 'bbox' 和 'confidence'
        iou_threshold: IoU 阈值
        
    返回:
        NMS 处理后的检测结果
    """
    if not detections:
        return []
    
    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while detections:
        # 取置信度最高的
        current = detections.pop(0)
        keep.append(current)
        
        # 移除与当前框 IoU 过高的框
        remaining = []
        for det in detections:
            iou = compute_iou(current['bbox'], det['bbox'])
            if iou < iou_threshold:
                remaining.append(det)
        
        detections = remaining
    
    return keep


def format_bbox(bbox: Tuple, format_type: str = 'pascal_voc') -> Dict:
    """格式转换工具
    
    支持格式:
        - pascal_voc: (x_min, y_min, x_max, y_max)
        - coco: (x_min, y_min, width, height)
        - yolo: (x_center, y_center, width, height) - 归一化
    """
    x_min, y_min, x_max, y_max = bbox
    
    result = {}
    result['pascal_voc'] = (x_min, y_min, x_max, y_max)
    result['coco'] = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    result['yolo'] = (x_center, y_center, width, height)
    
    return result[format_type]


class AverageMeter:
    """平均值计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # 测试 IoU 计算
    box1 = (0, 0, 100, 100)
    box2 = (50, 50, 150, 150)
    iou = compute_iou(box1, box2)
    print(f"IoU: {iou:.4f}")  # 应该是 0.1429
