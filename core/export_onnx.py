"""
ONNX 导出脚本 - 将 PyTorch 模型转换为 ONNX 格式
用于部署到推理引擎和边缘设备
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from ultralytics import YOLO


class ONNXExporter:
    """ONNX 模型导出器"""
    
    def __init__(self, weights_path: str, output_dir: str = './'):
        """初始化导出器
        
        参数:
            weights_path: PyTorch 权重文件路径 (.pt)
            output_dir: 输出目录
        """
        self.weights_path = Path(weights_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        
        print(f"✅ 初始化导出器")
        print(f"   权重: {weights_path}")
        print(f"   输出: {output_dir}")
    
    def export(
        self,
        opset_version: int = 12,
        simplify: bool = True,
        optimize_model: bool = True,
    ) -> str:
        """导出为 ONNX 格式
        
        参数:
            opset_version: ONNX OpSet 版本
            simplify: 是否使用 onnx-simplifier 简化模型
            optimize_model: 是否优化模型
            
        返回:
            导出的 ONNX 文件路径
        """
        print(f"\n🔄 开始导出 ONNX 模型...")
        
        # 加载模型
        model = YOLO(str(self.weights_path))
        
        # 设置输出路径
        onnx_path = self.output_dir / self.weights_path.stem + '.onnx'
        
        # 导出
        model.export(
            format='onnx',
            opset=opset_version,
            simplify=simplify,
        )
        
        # 官方导出通常在原路径附近，我们需要复制到目标位置
        import shutil
        # YOLO11 官方导出的文件
        default_onnx = self.weights_path.parent / (self.weights_path.stem + '.onnx')
        if default_onnx.exists() and default_onnx != onnx_path:
            shutil.copy(default_onnx, onnx_path)
        
        print(f"✅ 已导出 ONNX 模型: {onnx_path}")
        
        return str(onnx_path)
    
    def validate_onnx(self, onnx_path: str, test_image_shape: tuple = (1, 3, 640, 640)):
        """验证 ONNX 模型
        
        参数:
            onnx_path: ONNX 文件路径
            test_image_shape: 测试输入形状 (B, C, H, W)
        """
        if not ONNX_AVAILABLE:
            print("⚠️  onnxruntime 未安装，跳过验证")
            return
        
        print(f"\n✔️ 验证 ONNX 模型...")
        
        # 加载 ONNX 模型
        onnx_model = onnx.load(onnx_path)
        
        # 检查模型
        try:
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX 模型格式正确")
        except Exception as e:
            print(f"❌ ONNX 模型格式错误: {e}")
            return
        
        # 使用 onnxruntime 测试推理
        try:
            session = ort.InferenceSession(onnx_path)
            
            # 获取输入输出信息
            input_name = session.get_inputs()[0].name
            output_names = [o.name for o in session.get_outputs()]
            
            print(f"\n📊 模型信息:")
            print(f"   输入: {input_name} {session.get_inputs()[0].shape}")
            for output in session.get_outputs():
                print(f"   输出: {output.name} {output.shape}")
            
            # 执行测试推理
            test_input = np.random.randn(*test_image_shape).astype(np.float32)
            
            import time
            start = time.time()
            outputs = session.run(output_names, {input_name: test_input})
            elapsed = time.time() - start
            
            print(f"\n⚡ 推理性能:")
            print(f"   输入形状: {test_image_shape}")
            print(f"   推理时间: {elapsed*1000:.2f} ms")
            print(f"   输出数量: {len(outputs)}")
            
            print(f"\n✅ ONNX 模型验证通过!")
            
        except Exception as e:
            print(f"❌ ONNX 推理失败: {e}")
    
    def print_model_info(self, onnx_path: str):
        """打印模型详细信息"""
        if not ONNX_AVAILABLE:
            return
        
        onnx_model = onnx.load(onnx_path)
        graph = onnx_model.graph
        
        print(f"\n📋 ONNX 模型详细信息:")
        print(f"\n输入:")
        for input_tensor in graph.input:
            print(f"  - {input_tensor.name}: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
        
        print(f"\n输出:")
        for output_tensor in graph.output:
            print(f"  - {output_tensor.name}: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")
        
        print(f"\n算子数量: {len(graph.node)}")
        
        # 统计算子类型
        op_types = {}
        for node in graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
        
        print(f"\n算子类型统计:")
        for op_type, count in sorted(op_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  - {op_type}: {count}")
        
        # 模型大小
        import os
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"\n模型大小: {size_mb:.2f} MB")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ONNX 导出脚本')
    parser.add_argument('--weights', type=str, required=True, help='PyTorch 权重文件路径')
    parser.add_argument('--output', type=str, default='./', help='输出目录')
    parser.add_argument('--opset', type=int, default=12, help='ONNX OpSet 版本')
    parser.add_argument('--simplify', action='store_true', help='简化 ONNX 模型')
    parser.add_argument('--optimize', action='store_true', help='优化模型')
    parser.add_argument('--validate', action='store_true', help='验证导出的模型')
    parser.add_argument('--info', action='store_true', help='打印模型信息')
    
    args = parser.parse_args()
    
    # 创建导出器
    exporter = ONNXExporter(args.weights, args.output)
    
    # 导出
    onnx_path = exporter.export(
        opset_version=args.opset,
        simplify=args.simplify,
        optimize_model=args.optimize,
    )
    
    # 验证
    if args.validate:
        exporter.validate_onnx(onnx_path)
    
    # 打印信息
    if args.info:
        exporter.print_model_info(onnx_path)
    
    print(f"\n✅ 导出完成!")
    print(f"   ONNX 文件: {onnx_path}")


if __name__ == '__main__':
    main()
