# 获取yolo 默认的config_yaml内容并保存到指定路径
from pathlib import Path
from ultralytics import YOLO

import yaml
def save_default_config(save_path: Path):
    YOLO

if __name__ == "__main__":
    save_path = Path("config.yaml")
    save_default_config(save_path)