"""
训练界面 - TrainWidget
包含数据集选择、模型选择、参数配置和训练功能
"""

import os
import yaml
from pathlib import Path
import io
import sys
import re
from contextlib import redirect_stdout
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QLineEdit,
    QTextEdit,
    QFileDialog,
    QGroupBox,
    QGridLayout,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
    QProgressBar,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont

import sys

sys.path.append(str(Path(__file__).parent.parent))
from core.train import YoloDetector
from core.output_consumer import OutputConsumer, OutputConsumerManager
from core.output_logger import OutputLogger, TeeWriter
from gui.training_progress_consumer import TrainingProgressConsumer


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
        """注册输出消费者

        Args:
            consumer: OutputConsumer 实例
        """
        self.consumer_manager.register(consumer)

    def unregister_output_consumer(self, consumer: OutputConsumer) -> None:
        """注销输出消费者

        Args:
            consumer: 要注销的 OutputConsumer 实例
        """
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

            # 验证数据集
            from pathlib import Path

            dataset_path = Path(self.dataset_root)
            data_yaml = dataset_path / "data.yaml"

            if not data_yaml.exists():
                raise FileNotFoundError(f"data.yaml 不存在: {data_yaml}")

            # 验证images目录
            images_dir = dataset_path / "images"
            if not images_dir.exists():
                raise FileNotFoundError(f"images 目录不存在: {images_dir}")

            train_dir = images_dir / "train"
            val_dir = images_dir / "val"

            if not train_dir.exists():
                raise FileNotFoundError(f"images/train 目录不存在: {train_dir}")
            if not val_dir.exists():
                raise FileNotFoundError(f"images/val 目录不存在: {val_dir}")

            train_count = len(list(train_dir.glob("*.jpg"))) + len(
                list(train_dir.glob("*.png"))
            )
            val_count = len(list(val_dir.glob("*.jpg"))) + len(
                list(val_dir.glob("*.png"))
            )

            if train_count == 0:
                raise ValueError(f"训练目录中没有图片: {train_dir}")
            if val_count == 0:
                raise ValueError(f"验证目录中没有图片: {val_dir}")

            self.log_signal.emit(f"✅ 数据集验证通过")
            self.log_signal.emit(f"   - 训练图片: {train_count} 张")
            self.log_signal.emit(f"   - 验证图片: {val_count} 张\n")

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
        self.is_running = False


class TrainWidget(QWidget):
    """训练界面组件"""

    # 模型版本到配置目录的映射
    VERSION_CONFIG_DIR = {
        0: "yolopt/11",  # YOLO11
        1: "yolopt/9",  # YOLO9
        2: "yolopt/8",  # YOLO8
        3: "yolopt/12",  # YOLO12
    }

    def __init__(self):
        super().__init__()
        self.dataset_root = ""
        self.is_training = False
        self.train_thread = None
        self.progress_timer = None  # 用于轮询训练进度的计时器

        # 创建训练进度消费者
        self.progress_consumer = TrainingProgressConsumer()

        self._init_ui()
        self._load_config()
        self._setup_connections()

    def _init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ============== 第一行：按钮 ==============
        button_layout = QHBoxLayout()

        self.select_data_btn = QPushButton("选择数据集")
        self.select_data_btn.setMinimumHeight(40)
        self.select_data_btn.setToolTip("选择包含images和labels目录的数据集根目录")
        button_layout.addWidget(self.select_data_btn)

        self.train_btn = QPushButton("开始训练")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.setEnabled(False)
        button_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        # ============== 第二行：模型选择 ==============
        model_layout = QHBoxLayout()

        # 数据集路径显示
        model_layout.addWidget(QLabel("数据集:"))
        self.dataset_label = QLabel("未选择")
        self.dataset_label.setStyleSheet("color: #666; font-style: italic;")
        model_layout.addWidget(self.dataset_label, 1)

        model_layout.addSpacing(20)

        # 模型版本选择
        model_layout.addWidget(QLabel("YOLO版本:"))
        self.version_combo = QComboBox()
        self.version_combo.addItems(["YOLO11", "YOLO9", "YOLO8", "YOLO12"])
        self.version_combo.setCurrentIndex(0)  # 默认选择YOLO11
        self.version_combo.setMinimumWidth(100)
        model_layout.addWidget(self.version_combo)

        # 模型大小选择
        model_layout.addWidget(QLabel("模型大小:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["最快", "平衡", "精度高", "更高精度", "最高精度"])
        self.model_combo.setCurrentIndex(1)  # 默认选择small
        self.model_combo.setMinimumWidth(180)
        model_layout.addWidget(self.model_combo)

        model_layout.addStretch()

        main_layout.addLayout(model_layout)

        # ============== 第三行：参数配置（占满剩余空间） ==============
        # 使用分割器：左边参数配置，右边日志输出
        config_log_layout = QHBoxLayout()

        # 左边：参数配置
        config_group = QGroupBox("训练参数配置")
        config_layout = QGridLayout()
        config_layout.setSpacing(10)

        row = 0

        # Epochs
        config_layout.addWidget(QLabel("训练轮数 (Epochs):"), row, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(30)
        self.epochs_spin.setToolTip("训练的总轮数")
        config_layout.addWidget(self.epochs_spin, row, 1)
        row += 1

        # Batch Size
        config_layout.addWidget(QLabel("批次大小 (Batch Size):"), row, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(32)
        self.batch_spin.setToolTip("每批处理的图片数量")
        config_layout.addWidget(self.batch_spin, row, 1)
        row += 1

        # Learning Rate
        config_layout.addWidget(QLabel("学习率 (Learning Rate):"), row, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setToolTip("学习率，控制训练速度")
        config_layout.addWidget(self.lr_spin, row, 1)
        row += 1

        # Weight Decay
        config_layout.addWidget(QLabel("权重衰减 (Weight Decay):"), row, 0)
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0.0, 0.01)
        self.wd_spin.setDecimals(4)
        self.wd_spin.setSingleStep(0.0001)
        self.wd_spin.setValue(0.0005)
        self.wd_spin.setToolTip("防止过拟合的正则化参数")
        config_layout.addWidget(self.wd_spin, row, 1)
        row += 1

        # Warmup Epochs
        config_layout.addWidget(QLabel("预热轮数 (Warmup):"), row, 0)
        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 20)
        self.warmup_spin.setValue(5)
        self.warmup_spin.setToolTip("预热阶段的轮数")
        config_layout.addWidget(self.warmup_spin, row, 1)
        row += 1

        # Box Loss Weight
        config_layout.addWidget(QLabel("框损失权重 (Box Loss):"), row, 0)
        self.box_loss_spin = QDoubleSpinBox()
        self.box_loss_spin.setRange(0.1, 20.0)
        self.box_loss_spin.setDecimals(1)
        self.box_loss_spin.setSingleStep(0.5)
        self.box_loss_spin.setValue(7.5)
        config_layout.addWidget(self.box_loss_spin, row, 1)
        row += 1

        # Cls Loss Weight
        config_layout.addWidget(QLabel("分类损失权重 (Cls Loss):"), row, 0)
        self.cls_loss_spin = QDoubleSpinBox()
        self.cls_loss_spin.setRange(0.1, 10.0)
        self.cls_loss_spin.setDecimals(1)
        self.cls_loss_spin.setSingleStep(0.1)
        self.cls_loss_spin.setValue(1.5)
        config_layout.addWidget(self.cls_loss_spin, row, 1)
        row += 1

        # 输出目录
        config_layout.addWidget(QLabel("输出目录:"), row, 0)
        output_h_layout = QHBoxLayout()
        self.output_edit = QLineEdit("./results")
        output_h_layout.addWidget(self.output_edit)
        self.output_btn = QPushButton("浏览...")
        self.output_btn.setMaximumWidth(80)
        output_h_layout.addWidget(self.output_btn)
        config_layout.addLayout(output_h_layout, row, 1)
        row += 1

        config_layout.setRowStretch(row, 1)  # 剩余空间
        config_group.setLayout(config_layout)
        config_log_layout.addWidget(config_group, 2)  # 占2份

        # 右边：训练日志和进度显示
        log_group = QGroupBox("训练日志和进度")
        log_layout = QVBoxLayout()

        # 进度指标显示
        progress_widget = QWidget()
        progress_grid = QGridLayout()
        progress_grid.setSpacing(8)

        # 第一行：Epoch, GPU_mem, box_loss
        progress_grid.addWidget(QLabel("Epoch:"), 0, 0)
        self.epoch_label = QLabel("0/0")
        self.epoch_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        progress_grid.addWidget(self.epoch_label, 0, 1)

        progress_grid.addWidget(QLabel("GPU_mem:"), 0, 2)
        self.gpu_mem_label = QLabel("0.0 GB")
        self.gpu_mem_label.setStyleSheet("color: #00aa00; font-weight: bold;")
        progress_grid.addWidget(self.gpu_mem_label, 0, 3)

        progress_grid.addWidget(QLabel("box_loss:"), 0, 4)
        self.box_loss_label = QLabel("0.0000")
        self.box_loss_label.setStyleSheet("color: #ff6600; font-weight: bold;")
        progress_grid.addWidget(self.box_loss_label, 0, 5)

        # 第二行：cls_loss, dfl_loss, Instances
        progress_grid.addWidget(QLabel("cls_loss:"), 1, 0)
        self.cls_loss_label = QLabel("0.0000")
        self.cls_loss_label.setStyleSheet("color: #ff6600; font-weight: bold;")
        progress_grid.addWidget(self.cls_loss_label, 1, 1)

        progress_grid.addWidget(QLabel("dfl_loss:"), 1, 2)
        self.dfl_loss_label = QLabel("0.0000")
        self.dfl_loss_label.setStyleSheet("color: #ff6600; font-weight: bold;")
        progress_grid.addWidget(self.dfl_loss_label, 1, 3)

        progress_grid.addWidget(QLabel("Instances:"), 1, 4)
        self.instances_label = QLabel("0")
        self.instances_label.setStyleSheet("color: #9933ff; font-weight: bold;")
        progress_grid.addWidget(self.instances_label, 1, 5)

        # 第三行：Size
        progress_grid.addWidget(QLabel("Size:"), 2, 0)
        self.size_label = QLabel("640")
        self.size_label.setStyleSheet("color: #666666; font-weight: bold;")
        progress_grid.addWidget(self.size_label, 2, 1)

        progress_widget.setLayout(progress_grid)
        log_layout.addWidget(progress_widget)

        # 分隔线
        separator = QLabel("─" * 80)
        log_layout.addWidget(separator)

        # 训练日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        config_log_layout.addWidget(log_group, 3)  # 占3份

        main_layout.addLayout(config_log_layout, 1)  # 占满剩余空间

    def _validate_dataset(self, dataset_path):
        """验证数据集结构

        Returns:
            {
                'valid': bool,
                'errors': str,
                'train_count': int,
                'val_count': int,
                'test_count': int
            }
        """
        errors = []
        train_count = 0
        val_count = 0
        test_count = 0

        # 检查data.yaml
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            errors.append("❌ 缺少 data.yaml")

        # 检查images目录
        images_dir = dataset_path / "images"
        if not images_dir.exists():
            errors.append("❌ 缺少 images 目录")
        else:
            # 检查train
            train_dir = images_dir / "train"
            if not train_dir.exists():
                errors.append("❌ 缺少 images/train 目录")
            else:
                train_files = list(train_dir.glob("*"))
                train_count = len(
                    [
                        f
                        for f in train_files
                        if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
                    ]
                )
                if train_count == 0:
                    errors.append("⚠️  images/train 目录为空")

            # 检查val
            val_dir = images_dir / "val"
            if not val_dir.exists():
                errors.append("❌ 缺少 images/val 目录")
            else:
                val_files = list(val_dir.glob("*"))
                val_count = len(
                    [
                        f
                        for f in val_files
                        if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
                    ]
                )
                if val_count == 0:
                    errors.append("⚠️  images/val 目录为空")

            # 检查test（可选）
            test_dir = images_dir / "test"
            if test_dir.exists():
                test_files = list(test_dir.glob("*"))
                test_count = len(
                    [
                        f
                        for f in test_files
                        if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
                    ]
                )

        # 检查labels目录（可选，YOLO格式）
        labels_dir = dataset_path / "labels"
        if labels_dir.exists():
            if not (labels_dir / "train").exists():
                errors.append("⚠️  labels/train 目录缺少")
            if not (labels_dir / "val").exists():
                errors.append("⚠️  labels/val 目录缺少")

        valid = not any("❌" in e for e in errors)

        return {
            "valid": valid,
            "errors": "\n".join(errors) if errors else "✅ 数据集结构完整",
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
        }

    def _load_config(self):
        """加载配置文件"""
        try:
            config_path = self._get_config_path()
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                # 加载训练参数
                if "training" in config:
                    self.epochs_spin.setValue(config["training"].get("epochs", 30))
                    self.batch_spin.setValue(config["training"].get("batch", 32))
                    self.lr_spin.setValue(
                        config["training"].get("lr0", 0.001)
                    )
                    self.wd_spin.setValue(
                        config["training"].get("weight_decay", 0.0005)
                    )
                    self.warmup_spin.setValue(
                        config["training"].get("warmup_epochs", 5)
                    )
                    self.box_loss_spin.setValue(
                        config["training"].get("box", 7.5)
                    )
                    self.cls_loss_spin.setValue(
                        config["training"].get("cls", 1.5)
                    )

                self.log(f"✅ 配置文件加载成功: {config_path}")
            else:
                self.log(f"⚠️ 配置文件不存在: {config_path}")
        except Exception as e:
            self.log(f"⚠️ 加载配置文件失败: {e}")

    def _save_config(self):
        """保存配置到文件"""
        try:
            config_path = self._get_config_path()
            # 读取现有配置
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 更新模型配置
            version_names = ["yolo11", "yolo9", "yolo8", "yolo12"]
            size_names = ["nano", "small", "medium", "large", "xlarge"]
            config["model"]["version"] = version_names[self.version_combo.currentIndex()]
            config["model"]["backbone"] = size_names[self.model_combo.currentIndex()]

            # 更新训练参数（使用官方YOLO参数名）
            config["training"]["epochs"] = self.epochs_spin.value()
            config["training"]["batch"] = self.batch_spin.value()
            config["training"]["lr0"] = self.lr_spin.value()
            config["training"]["weight_decay"] = self.wd_spin.value()
            config["training"]["warmup_epochs"] = self.warmup_spin.value()
            config["training"]["box"] = self.box_loss_spin.value()
            config["training"]["cls"] = self.cls_loss_spin.value()

            # 保存
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            self.log(f"✅ 配置已保存到: {config_path}")
            return True
        except Exception as e:
            self.log(f"❌ 保存配置失败: {e}")
            return False

    def _get_config_path(self):
        """获取配置文件路径

        Returns:
            str: 固定配置文件路径
        """
        return "configs/config.yaml"

    def _setup_connections(self):
        """设置信号连接"""
        self.select_data_btn.clicked.connect(self.select_dataset)
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.output_btn.clicked.connect(self.select_output_dir)
        # 模型版本或大小改变时，重新加载对应的配置文件
        self.version_combo.currentIndexChanged.connect(self.on_model_changed)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

    def select_dataset(self):
        """选择数据集目录"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择数据集根目录",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if directory:
            # 验证数据集结构
            dataset_path = Path(directory)
            data_yaml = dataset_path / "data.yaml"

            if not data_yaml.exists():
                QMessageBox.warning(
                    self,
                    "数据集错误",
                    f"所选目录中未找到 data.yaml 文件。\n\n"
                    f"请确保数据集目录包含:\n"
                    f"  - data.yaml (数据集配置)\n"
                    f"  - images/train/ (训练图片)\n"
                    f"  - images/val/ (验证图片)",
                )
                return

            # 详细验证数据集结构
            validation_result = self._validate_dataset(dataset_path)
            if not validation_result["valid"]:
                QMessageBox.warning(
                    self,
                    "数据集验证失败",
                    f"数据集结构不完整:\n\n{validation_result['errors']}",
                )
                return

            self.dataset_root = directory
            self.dataset_label.setText(directory)
            self.dataset_label.setStyleSheet("color: #000;")
            self.train_btn.setEnabled(True)
            self.log(f"✅ 已选择数据集: {directory}")
            self.log(f"   - 训练图片: {validation_result['train_count']} 张")
            self.log(f"   - 验证图片: {validation_result['val_count']} 张")
            if validation_result["test_count"] > 0:
                self.log(f"   - 测试图片: {validation_result['test_count']} 张")

    def select_output_dir(self):
        """选择输出目录"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            self.output_edit.text(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if directory:
            self.output_edit.setText(directory)
            self.log(f"✅ 输出目录: {directory}")

    def start_training(self):
        """开始训练"""
        if not self.dataset_root:
            QMessageBox.warning(self, "错误", "请先选择数据集！")
            return

        # 保存配置
        if not self._save_config():
            # 配置保存失败，取消训练
            return

        # 更新UI状态
        self.is_training = True
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.select_data_btn.setEnabled(False)
        self.version_combo.setEnabled(False)  # 训练中禁用版本选择
        self.model_combo.setEnabled(False)  # 训练中禁用模型选择

        # 保持参数输入框可用，允许用户在训练过程中实时调整参数（下次训练生效）
        self.epochs_spin.setEnabled(True)
        self.batch_spin.setEnabled(True)
        self.lr_spin.setEnabled(True)
        self.wd_spin.setEnabled(True)
        self.warmup_spin.setEnabled(True)
        self.box_loss_spin.setEnabled(True)
        self.cls_loss_spin.setEnabled(True)
        self.output_edit.setEnabled(True)
        self.output_btn.setEnabled(True)

        self.log_text.clear()

        # 重置进度显示和消费者
        self.epoch_label.setText("0/0")
        self.gpu_mem_label.setText("0.0 GB")
        self.box_loss_label.setText("0.0000")
        self.cls_loss_label.setText("0.0000")
        self.dfl_loss_label.setText("0.0000")
        self.instances_label.setText("0")
        self.size_label.setText("640")

        # 创建并启动训练线程
        config_path = self._get_config_path()
        output_dir = self.output_edit.text()
        self.train_thread = TrainThread(config_path, self.dataset_root, output_dir)
        self.train_thread.log_signal.connect(self.log)
        self.train_thread.finished_signal.connect(self.on_training_finished)

        # 注册进度消费者
        self.progress_consumer.on_training_start()  # 重置消费者状态
        self.train_thread.register_output_consumer(self.progress_consumer)

        self.train_thread.start()

        # 启动进度轮询定时器（每500毫秒更新一次UI）
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_training_progress)
        self.progress_timer.start(500)

    def update_training_progress(self):
        """轮询更新训练进度UI

        从消费者获取已解析的进度数据，更新UI标签
        """
        if not self.train_thread:
            return

        try:
            # 从消费者获取最新的进度数据
            progress_data = self.progress_consumer.get_progress_data()

            # 更新 Epoch
            epoch = progress_data.get("epoch")
            total_epochs = progress_data.get("total_epochs")
            if epoch is not None and total_epochs is not None:
                self.epoch_label.setText(f"{epoch}/{total_epochs}")

            # 更新 GPU 内存
            gpu_mem = progress_data.get("gpu_mem")
            if gpu_mem is not None:
                self.gpu_mem_label.setText(f"{gpu_mem:.2f} GB")

            # 更新 Box Loss
            box_loss = progress_data.get("box_loss")
            if box_loss is not None:
                self.box_loss_label.setText(f"{box_loss:.4f}")

            # 更新 Cls Loss
            cls_loss = progress_data.get("cls_loss")
            if cls_loss is not None:
                self.cls_loss_label.setText(f"{cls_loss:.4f}")

            # 更新 DFL Loss
            dfl_loss = progress_data.get("dfl_loss")
            if dfl_loss is not None:
                self.dfl_loss_label.setText(f"{dfl_loss:.4f}")

            # 更新 Instances
            instances = progress_data.get("instances")
            if instances is not None:
                self.instances_label.setText(f"{instances:.0f}")

            # 更新 Size
            size = progress_data.get("size")
            if size is not None:
                self.size_label.setText(f"{size}")

            # 通知消费者处理新输出
            if self.train_thread.output_buffer:
                captured_output = self.train_thread.output_buffer.getvalue()
                # 消费者内部会处理增量部分
                self.progress_consumer.consume(captured_output)

        except Exception as e:
            # 静默处理异常，不中断轮询
            pass

    def stop_training(self):
        """停止训练"""
        if self.train_thread and self.train_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认停止",
                "确定要停止训练吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self.log("\n⚠️ 正在停止训练...")
                self.train_thread.stop()
                self.train_thread.wait()
                self.on_training_finished(False, "用户取消")

    def on_model_changed(self, index):
        """模型选择改变时的处理函数"""
        if not self.is_training:
            # 只在不训练时重新加载配置
            self.log(f"📋 模型已改变，加载对应的配置文件...")
            self._load_config()

    def on_training_finished(self, success, message):
        """训练完成回调"""
        # 停止进度轮询定时器
        if self.progress_timer:
            self.progress_timer.stop()
            self.progress_timer = None

        self.is_training = False
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.select_data_btn.setEnabled(True)
        self.version_combo.setEnabled(True)  # 训练完成后重新启用版本选择
        self.model_combo.setEnabled(True)  # 训练完成后重新启用模型选择

        # 确保所有参数输入框保持启用
        self.epochs_spin.setEnabled(True)
        self.batch_spin.setEnabled(True)
        self.lr_spin.setEnabled(True)
        self.wd_spin.setEnabled(True)
        self.warmup_spin.setEnabled(True)
        self.box_loss_spin.setEnabled(True)
        self.cls_loss_spin.setEnabled(True)
        self.output_edit.setEnabled(True)
        self.output_btn.setEnabled(True)

        # 清理线程
        if self.train_thread:
            self.train_thread.quit()
            self.train_thread.wait()
            self.train_thread = None

        if success:
            QMessageBox.information(self, "训练完成", "模型训练已完成！")
        else:
            QMessageBox.warning(self, "训练失败", f"训练失败: {message}")

    def log(self, message):
        """添加日志"""
        self.log_text.append(message)
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
