"""
YOLO11 模型下载脚本

下载所有 YOLO11 预训练模型到 yolopt/11 目录
支持的模型规格：nano, small, medium, large, xlarge
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO


def setup_yolopt_directory():
    """创建 yolopt/11 目录结构"""
    yolopt_dir = Path('yolopt') / '11'
    yolopt_dir.mkdir(parents=True, exist_ok=True)
    return yolopt_dir


def download_yolo11_models(yolopt_dir: Path):
    """下载所有 YOLO11 模型
    
    Args:
        yolopt_dir: 模型保存目录
    """
    # 支持的模型规格
    models = {
        'nano': 'yolo11n.pt',
        'small': 'yolo11s.pt',
        'medium': 'yolo11m.pt',
        'large': 'yolo11l.pt',
        'xlarge': 'yolo11x.pt',
    }
    
    print("=" * 60)
    print("YOLO11 模型下载工具")
    print("=" * 60)
    print(f"\n📁 模型保存目录: {yolopt_dir.resolve()}\n")
    
    # 设置环保存目录（YOLO 会自动使用这个位置）
    os.environ['YOLO_HOME'] = str(yolopt_dir.parent)  # 父目录是 yolopt
    
    success_count = 0
    fail_count = 0
    
    for size, model_name in models.items():
        try:
            print(f"[{size.upper()}] 下载 {model_name}...", end=' ')
            
            # 加载模型会自动下载到指定目录
            model = YOLO(model_name)
            
            # 模型文件路径
            model_path = yolopt_dir / model_name
            
            if model_path.exists():
                print(f"✅ 完成 ({model_path.stat().st_size / 1024 / 1024:.1f}MB)")
                success_count += 1
            else:
                print(f"⚠️  未找到模型文件")
                fail_count += 1
        
        except Exception as e:
            print(f"❌ 失败 - {str(e)}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"下载完成: {success_count} 个成功, {fail_count} 个失败")
    print("=" * 60)
    
    # 显示下载的模型列表
    print(f"\n📋 已保存的模型:")
    if yolopt_dir.exists():
        for model_file in sorted(yolopt_dir.glob('yolo11*.pt')):
            size_mb = model_file.stat().st_size / 1024 / 1024
            print(f"  ✓ {model_file.name:20s} ({size_mb:6.1f}MB)")
    
    return success_count > 0


def verify_models(yolopt_dir: Path):
    """验证模型完整性
    
    Args:
        yolopt_dir: 模型目录
    """
    print(f"\n🔍 验证模型...\n")
    
    models_to_check = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']
    
    for model_name in models_to_check:
        model_path = yolopt_dir / model_name
        if model_path.exists():
            try:
                model = YOLO(str(model_path))
                print(f"  ✅ {model_name}: 可用")
            except Exception as e:
                print(f"  ❌ {model_name}: 损坏 - {e}")
        else:
            print(f"  ⚠️  {model_name}: 未找到")


if __name__ == '__main__':
    try:
        # 1. 创建目录
        print("📁 创建目录结构...\n")
        yolopt_dir = setup_yolopt_directory()
        print(f"✅ 目录创建成功: {yolopt_dir.resolve()}\n")
        
        # 2. 下载模型
        print("📥 开始下载模型...\n")
        success = download_yolo11_models(yolopt_dir)
        
        # 3. 验证模型
        if success:
            verify_models(yolopt_dir)
        
        # 4. 显示使用说明
        print("\n" + "=" * 60)
        print("✅ 完成！现在可以开始训练了")
        print("=" * 60)
        print(f"\n💡 模型位置: {yolopt_dir.resolve()}")
        print(f"💡 配置文件: config.yaml")
        print(f"💡 运行训练: python app.py")
        print()
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
