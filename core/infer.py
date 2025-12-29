"""
推理脚本 - 支持单张/文件夹推理、可视化、结果导出
支持 PyTorch 和 ONNX 推理
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import torch
from dataclasses import dataclass
import json
from datetime import datetime

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from ultralytics import YOLO


@dataclass
class DetectionResult:
    """检测结果数据类"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    
    def to_dict(self):
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'bbox': list(self.bbox),
        }


class DefectInference:
    """工业缺陷检测推理器"""
    
    def __init__(
        self,
        weights: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'cuda',
        use_onnx: bool = False,
    ):
        """初始化推理器
        
        参数:
            weights: 权重文件路径 (.pt 或 .onnx)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            device: 设备 ('cuda' 或 'cpu')
            use_onnx: 是否使用 ONNX 推理
        """
        self.weights = Path(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # 类别信息
        self.class_names = {0: 'person'}
        self.num_classes = 1
        
        # 加载模型
        if use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
        
        print(f"✅ 模型已加载: {weights}")
    
    def _load_pytorch_model(self):
        """加载 PyTorch 模型"""
        self.model = YOLO(str(self.weights))
        self.inference_type = 'pytorch'
    
    def _load_onnx_model(self):
        """加载 ONNX 模型"""
        if not ONNX_AVAILABLE:
            raise ImportError("Please install onnxruntime: pip install onnxruntime")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
            if self.device == 'cuda' else ['CPUExecutionProvider']
        
        self.ort_session = ort.InferenceSession(
            str(self.weights),
            providers=providers
        )
        
        # 获取输入输出信息
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [o.name for o in self.ort_session.get_outputs()]
        self.input_shape = self.ort_session.get_inputs()[0].shape
        
        self.inference_type = 'onnx'
        print(f"✅ ONNX 模型已加载 (输入: {self.input_shape})")
    
    def infer(self, image: np.ndarray) -> List[DetectionResult]:
        """对单张图像进行推理
        
        参数:
            image: 输入图像 (BGR格式, HxWx3)
            
        返回:
            检测结果列表
        """
        if self.inference_type == 'pytorch':
            return self._infer_pytorch(image)
        else:
            return self._infer_onnx(image)
    
    def _infer_pytorch(self, image: np.ndarray) -> List[DetectionResult]:
        """PyTorch 推理"""
        # YOLO11 API 直接推理
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        print(results[0].boxes)
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                detections.append(DetectionResult(
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, 'unknown'),
                    confidence=conf,
                    bbox=(x_min, y_min, x_max, y_max),
                ))
        
        return detections
    
    def _infer_onnx(self, image: np.ndarray) -> List[DetectionResult]:
        """ONNX 推理"""
        # 预处理
        img_resized = cv2.resize(image, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        # 推理
        outputs = self.ort_session.run(self.output_names, {self.input_name: img_batch})
        
        # 后处理 (YOLO11 ONNX 输出格式)
        predictions = outputs[0][0]  # (num_detections, 6)
        
        # 恢复到原始图像坐标
        h_orig, w_orig = image.shape[:2]
        h_resized, w_resized = 640, 640
        scale_x = w_orig / w_resized
        scale_y = h_orig / h_resized
        
        detections = []
        for pred in predictions:
            x_center, y_center, width, height, conf = pred[:5]
            cls_id = int(pred[5])
            
            if conf < self.conf_threshold:
                continue
            
            # 从中心坐标转换为角坐标
            x_min = int((x_center - width / 2) * scale_x)
            y_min = int((y_center - height / 2) * scale_y)
            x_max = int((x_center + width / 2) * scale_x)
            y_max = int((y_center + height / 2) * scale_y)
            
            # 边界约束
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w_orig, x_max)
            y_max = min(h_orig, y_max)
            
            detections.append(DetectionResult(
                class_id=cls_id,
                class_name=self.class_names.get(cls_id, 'unknown'),
                confidence=float(conf),
                bbox=(x_min, y_min, x_max, y_max),
            ))
        
        return detections
    
    def infer_batch(self, image_paths: List[str]) -> Dict[str, List[DetectionResult]]:
        """批量推理
        
        参数:
            image_paths: 图像路径列表
            
        返回:
            {图像路径: 检测结果}
        """
        results = {}
        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️  无法读取图像: {img_path}")
                continue
            
            detections = self.infer(image)
            results[img_path] = detections
        
        return results
    
    def visualize(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """可视化检测结果
        
        参数:
            image: 原始图像
            detections: 检测结果列表
            save_path: 保存路径 (可选)
            
        返回:
            可视化后的图像
        """
        vis_image = image.copy()
        
        # 颜色
        color = (0, 255, 0)  # 绿色 (BGR)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        for det in detections:
            x_min, y_min, x_max, y_max = det.bbox
            
            # 绘制边框
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, thickness)
            
            # 绘制标签
            label = f"{det.class_name} {det.confidence:.2f}"
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = x_min
            text_y = y_min - 5
            
            # 绘制文字背景
            cv2.rectangle(
                vis_image,
                (text_x, text_y - text_size[1] - 4),
                (text_x + text_size[0], text_y),
                color,
                -1
            )
            
            # 绘制文字
            cv2.putText(
                vis_image,
                label,
                (text_x, text_y - 2),
                font,
                font_scale,
                (255, 255, 255),  # 白色文字
                font_thickness
            )
        
        # 保存
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"✅ 可视化结果已保存: {save_path}")
        
        return vis_image
    
    def export_results(self, results: Dict, save_path: str):
        """导出推理结果为 JSON
        
        参数:
            results: {图像路径: 检测结果}
            save_path: 保存路径
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'results': {}
        }
        
        for img_path, detections in results.items():
            export_data['results'][img_path] = [det.to_dict() for det in detections]
        
        with open(save_path, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 结果已导出: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='缺陷检测推理脚本')
    parser.add_argument('--weights', type=str, required=True, help='权重文件路径')
    parser.add_argument('--image', type=str, default=None, help='单张图像路径')
    parser.add_argument('--folder', type=str, default=None, help='图像文件夹路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--onnx', action='store_true', help='使用 ONNX 推理')
    parser.add_argument('--save-vis', type=str, default=None, help='保存可视化目录')
    parser.add_argument('--save-json', type=str, default=None, help='保存JSON结果文件')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = DefectInference(
        weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        use_onnx=args.onnx,
    )
    
    # 准备输出目录
    if args.save_vis:
        Path(args.save_vis).mkdir(parents=True, exist_ok=True)
    
    # 推理
    if args.image:
        # 单张图像
        print(f"\n🖼️  推理单张图像: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"❌ 无法读取图像: {args.image}")
            return
        
        detections = inferencer.infer(image)
        
        print(f"✅ 检测到 {len(detections)} 个目标:")
        for det in detections:
            print(f"  - {det.class_name}: {det.confidence:.2f} {det.bbox}")
        
        # 可视化
        if args.save_vis:
            vis_image = inferencer.visualize(
                image,
                detections,
                os.path.join(args.save_vis, Path(args.image).stem + '_vis.jpg')
            )
    
    elif args.folder:
        # 文件夹推理
        print(f"\n📁 推理文件夹: {args.folder}")
        image_dir = Path(args.folder)
        image_files = sorted(
            list(image_dir.glob('*.jpg')) +
            list(image_dir.glob('*.png')) +
            list(image_dir.glob('*.bmp'))
        )
        
        print(f"📊 找到 {len(image_files)} 张图像")
        
        all_results = {}
        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            detections = inferencer.infer(image)
            all_results[str(img_path)] = detections
            
            print(f"✅ {img_path.name}: {len(detections)} 个目标")
            
            # 可视化
            if args.save_vis:
                vis_image = inferencer.visualize(
                    image,
                    detections,
                    os.path.join(args.save_vis, img_path.stem + '_vis.jpg')
                )
        
        # 导出结果
        if args.save_json:
            inferencer.export_results(all_results, args.save_json)
    
    else:
        print("❌ 请指定 --image 或 --folder")


if __name__ == '__main__':
    main()
