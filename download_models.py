"""
YOLO 模型下载脚本

下载所有 YOLO 预训练模型到 yolopt/{version} 目录
支持版本: YOLO11, YOLO9, YOLO8, YOLO26
每个版本支持的模型规格：nano, small, medium, large, xlarge (部分版本)
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO


def setup_yolopt_directory():
    """创建 yolopt 目录结构"""
    yolopt_dir = Path('yolopt')
    yolopt_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建各版本子目录
    versions = ['11', '9', '8', '26']
    for version in versions:
        version_dir = yolopt_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
    
    return yolopt_dir


def download_models(yolopt_dir: Path):
    """下载所有 YOLO 版本的模型
    
    Args:
        yolopt_dir: yolopt 保存目录
    """
    # 定义所有模型版本
    model_versions = {
        '11': {
            'type': 'yolo11',
            'models': {
                'nano': 'yolo11n.pt',
                'small': 'yolo11s.pt',
                'medium': 'yolo11m.pt',
                'large': 'yolo11l.pt',
                'xlarge': 'yolo11x.pt',
            }
        },
        '9': {
            'type': 'yolo9',
            'models': {
                'nano': 'yolov9t.pt',
                'small': 'yolov9s.pt',
                'medium': 'yolov9m.pt',
                'large': 'yolov9c.pt',
                'xlarge': 'yolov9e.pt',
            }
        },
        '8': {
            'type': 'yolo8',
            'models': {
                'nano': 'yolov8n.pt',
                'small': 'yolov8s.pt',
                'medium': 'yolov8m.pt',
                'large': 'yolov8l.pt',
                'xlarge': 'yolov8x.pt',
            }
        },
        '12': {
            'type': 'yolo12',
            'models': {
                'nano': 'yolo12n.pt',
                'small': 'yolo12s.pt',
                'medium': 'yolo12m.pt',
                'large': 'yolo12l.pt',
                'xlarge': 'yolo12x.pt',                 
            }
        },
    }
    
    print("=" * 60)
    print("YOLO 模型下载工具 (支持YOLO11, YOLO9, YOLO8, YOLO12)")
    print("=" * 60)
    print(f"\n📁 模型保存目录: {yolopt_dir.resolve()}\n")
    
    # 设置环保存目录（YOLO 会自动使用这个位置）
    os.environ['YOLO_HOME'] = str(yolopt_dir)
    
    total_success = 0
    total_fail = 0
    
    for version, version_info in model_versions.items():
        print(f"\n{'='*60}")
        print(f"YOLO{version} 模型下载")
        print(f"{'='*60}")
        
        version_dir = yolopt_dir / version
        version_success = 0
        version_fail = 0
        
        for size, model_name in version_info['models'].items():
            try:
                print(f"[{size.upper():6s}] 下载 {model_name:15s}...", end=' ', flush=True)
                
                # 加载模型会自动下载到指定目录
                model = YOLO(model_name)
                
                # 查找下载的模型文件
                model_file = None
                # 首先检查标准位置
                for ext in ['.pt', '']:
                    check_path = version_dir / model_name
                    if check_path.exists():
                        model_file = check_path
                        break
                
                if model_file and model_file.exists():
                    size_mb = model_file.stat().st_size / 1024 / 1024
                    print(f"✅ 完成 ({size_mb:6.1f}MB)")
                    version_success += 1
                    total_success += 1
                else:
                    # 检查YOLO默认目录
                    yolo_home = Path(os.environ.get('YOLO_HOME', '~/.yolo')).expanduser()
                    alt_path = yolo_home / 'weights' / model_name
                    if alt_path.exists():
                        size_mb = alt_path.stat().st_size / 1024 / 1024
                        print(f"✅ 完成 ({size_mb:6.1f}MB) [在YOLO目录]")
                        version_success += 1
                        total_success += 1
                    else:
                        print(f"⚠️  未找到模型文件")
                        version_fail += 1
                        total_fail += 1
            
            except Exception as e:
                print(f"❌ 失败 - {str(e)[:40]}")
                version_fail += 1
                total_fail += 1
        
        print(f"\nYOLO{version}: {version_success} 个成功, {version_fail} 个失败")
    
    print("\n" + "=" * 60)
    print(f"总计: {total_success} 个成功, {total_fail} 个失败")
    print("=" * 60)
    
    # 显示下载的模型总结
    print(f"\n📋 已保存的模型汇总:")
    for version in ['11', '9', '8', '12']:
        version_dir = yolopt_dir / version
        pt_files = list(version_dir.glob('*.pt'))
        if pt_files:
            print(f"\n  YOLO{version}:")
            for model_file in sorted(pt_files):
                size_mb = model_file.stat().st_size / 1024 / 1024
                print(f"    ✓ {model_file.name:20s} ({size_mb:6.1f}MB)")
    
    return total_success > 0


def verify_models(yolopt_dir: Path):
    """验证模型完整性
    
    Args:
        yolopt_dir: 模型目录
    """
    print(f"\n🔍 验证模型...\n")
    
    verification_models = {
        '11': ['yolo11n.pt', 'yolo11s.pt'],
        '9': ['yolov9t.pt', 'yolov9s.pt'],
        '8': ['yolov8n.pt', 'yolov8s.pt'],
        '12': ['yolo12n.pt', 'yolo12s.pt'],
    }
    
    for version, models_list in verification_models.items():
        version_dir = yolopt_dir / version
        print(f"YOLO{version}:")
        for model_name in models_list:
            model_path = version_dir / model_name
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
        success = download_models(yolopt_dir)
        
        # 3. 验证模型
        if success:
            verify_models(yolopt_dir)
        
        # 4. 显示使用说明
        print("\n" + "=" * 60)
        print("✅ 完成！现在可以开始训练了")
        print("=" * 60)
        print(f"\n💡 模型位置: {yolopt_dir.resolve()}")
        print(f"💡 配置文件: yolopt/{{version}}/config_{{size}}.yaml")
        print(f"   - YOLO11: yolopt/11/config_{{nano|small|medium|large|xlarge}}.yaml")
        print(f"   - YOLO9:  yolopt/9/config_{{nano|small|medium|large|xlarge}}.yaml")
        print(f"   - YOLO8:  yolopt/8/config_{{nano|small|medium|large|xlarge}}.yaml")
        print(f"   - YOLO12: yolopt/12/config_{{nano|small|medium|large|xlarge}}.yaml")
        print(f"💡 运行训练: python app.py")
        print()
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
