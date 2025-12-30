"""
训练线程 - 单独文件
负责数据集验证、模型初始化和训练启动。
"""

import io
import sys
from pathlib import Path
import yaml
from PySide6.QtCore import QThread, Signal

from core.train import YoloDetector
from core.output_consumer import OutputConsumer, OutputConsumerManager
from core.output_logger import OutputLogger


class TrainThread(QThread):
    """训练线程"""

    log_signal = Signal(str)
    finished_signal = Signal(bool, str)  # success, message

    def __init__(self, config_path, dataset_root, output_dir):
        super().__init__()
        self.config_path = config_path
        self.dataset_root = dataset_root
        self.output_dir = output_dir
        self.is_running = True
        self.trainer = None  # 用于存储 trainer 引用
        self.output_buffer = io.StringIO()  # 用于捕获 print 输出

        # 输出消费者管理器
        self.consumer_manager = OutputConsumerManager()

    def register_output_consumer(self, consumer: OutputConsumer) -> None:
        """注册输出消费者"""
        self.consumer_manager.register(consumer)

    def unregister_output_consumer(self, consumer: OutputConsumer) -> None:
        """注销输出消费者"""
        self.consumer_manager.unregister(consumer)

    def run(self):
        """执行训练"""
        try:
            self.log_signal.emit("=" * 60)
            self.log_signal.emit("开始训练...")
            self.log_signal.emit(f"配置文件: {self.config_path}")
            self.log_signal.emit(f"数据集: {self.dataset_root}")
            self.log_signal.emit(f"输出目录: {self.output_dir}")
            self.log_signal.emit("=" * 60 + "\n")

            # 验证数据集（根据 data.yaml 路径解析）
            dataset_path = Path(self.dataset_root)
            data_yaml = dataset_path / "data.yaml"

            if not data_yaml.exists():
                raise FileNotFoundError(f"data.yaml 不存在: {data_yaml}")

            try:
                with open(data_yaml, "r", encoding="utf-8") as f:
                    data_cfg = yaml.safe_load(f) or {}
            except Exception as exc:
                raise ValueError(f"解析 data.yaml 失败: {exc}") from exc

            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_cfg = yaml.safe_load(f) or {}
                task = user_cfg.get("task", "detect")
            except Exception:
                task = "detect"

            base_path_val = data_cfg.get("path", ".")
            data_dir = data_yaml.parent

            def _resolve_path(entry):
                p = Path(entry)
                return p if p.is_absolute() else (data_dir / base_path_val / p).resolve()

            def _count_images(folder: Path):
                if not folder.exists():
                    return 0
                return len(
                    [
                        img
                        for img in folder.rglob("*")
                        if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
                    ]
                )

            train_entry = data_cfg.get("train")
            val_entry = data_cfg.get("val")
            if not train_entry or not val_entry:
                raise ValueError("data.yaml 缺少 train/val 定义")

            train_dir = _resolve_path(train_entry)
            val_dir = _resolve_path(val_entry)

            if not train_dir.exists():
                raise FileNotFoundError(f"训练集不存在: {train_dir}")
            if not val_dir.exists():
                raise FileNotFoundError(f"验证集不存在: {val_dir}")

            train_count = _count_images(train_dir)
            val_count = _count_images(val_dir)

            if train_count == 0:
                raise ValueError(f"训练目录中没有图片: {train_dir}")
            if val_count == 0:
                raise ValueError(f"验证目录中没有图片: {val_dir}")

            self.log_signal.emit(f"✅ 数据集验证通过")
            self.log_signal.emit(f"   - 训练图片: {train_count} 张")
            self.log_signal.emit(f"   - 验证图片: {val_count} 张\n")
            self.log_signal.emit(f"   - 任务: {task}\n")

            # 创建训练器
            self.log_signal.emit("📊 初始化模型...")
            self.trainer = YoloDetector(self.config_path, self.output_dir)

            # 开始训练，同时捕获输出并记录日志
            self.log_signal.emit("\n🚀 开始训练...\n")

            # 创建分流写入器：写到StringIO（用于消费者解析）和日志文件
            tee_writer = OutputLogger.create_tee_writer(
                string_buffer=self.output_buffer, log_dir=Path(self.output_dir)
            )

            # 将stdout重定向到TeeWriter
            old_stdout = sys.stdout
            sys.stdout = tee_writer

            try:
                results = self.trainer.train(self.dataset_root)
            finally:
                # 恢复stdout
                sys.stdout = old_stdout
                tee_writer.flush()

            # 训练完成，通知所有消费者
            self.consumer_manager.notify_training_end()

            if results:
                self.log_signal.emit("\n" + "=" * 60)
                self.log_signal.emit("✅ 训练完成！")
                self.log_signal.emit("=" * 60)
                self.finished_signal.emit(True, "训练完成")
            else:
                self.finished_signal.emit(False, "训练失败")

        except FileNotFoundError as e:
            error_msg = f"文件或目录未找到: {str(e)}"
            self.log_signal.emit(f"\n❌ {error_msg}")
            self.log_signal.emit("\n数据集应该包含以下结构:")
            self.log_signal.emit("dataset/")
            self.log_signal.emit("├── data.yaml")
            self.log_signal.emit("└── images/")
            self.log_signal.emit("    ├── train/  (训练图片)")
            self.log_signal.emit("    └── val/    (验证图片)")
            self.finished_signal.emit(False, error_msg)
        except ValueError as e:
            error_msg = f"数据验证失败: {str(e)}"
            self.log_signal.emit(f"\n❌ {error_msg}")
            self.finished_signal.emit(False, error_msg)
        except Exception as e:
            error_msg = f"训练出错: {str(e)}"
            self.log_signal.emit(f"\n❌ {error_msg}")
            import traceback

            self.log_signal.emit(f"\n详细信息:\n{traceback.format_exc()}")
            self.finished_signal.emit(False, error_msg)

    def stop(self):
        """停止训练"""
        # 若上层调用时 trainer 已创建，尝试保存并标记停止
        try:
            if self.trainer:
                if hasattr(self.trainer, "save_model"):
                    self.trainer.save_model()
                if hasattr(self.trainer, "stop"):
                    self.trainer.stop = True
        except Exception:
            pass
        self.is_running = False
