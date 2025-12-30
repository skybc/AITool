"""
训练界面 - TrainWidget
包含数据集选择、模型选择、参数配置和训练功能
"""

import os
import yaml
from pathlib import Path
from datetime import datetime
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
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

import sys

sys.path.append(str(Path(__file__).parent.parent))
from gui.train_thread import TrainThread
from gui.training_progress_consumer import TrainingProgressConsumer


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

        # 训练耗时跟踪
        self.training_start_time = None
        self.last_epoch_number = None
        self.last_epoch_timestamp = None
        self.epoch_time_accum = 0.0
        self.epoch_count = 0
        self.total_epochs_seen = None

        # 任务选项 (显示文本, YOLO task 名)
        self.task_options = [
            ("检测", "detect"),
            ("分割", "segment"),
            ("分类", "classify"),
            ("姿势估计", "pose"),
            ("定向检测", "obb"),
        ]

        self._init_ui()
        self._load_config()
        self._setup_connections()
        self._update_task_combo_state()

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

        # 任务类型选择
        model_layout.addSpacing(12)
        model_layout.addWidget(QLabel("任务:"))
        self.task_combo = QComboBox()
        for label, value in self.task_options:
            self.task_combo.addItem(label, userData=value)
        self.task_combo.setCurrentIndex(0)
        self.task_combo.setMinimumWidth(120)
        model_layout.addWidget(self.task_combo)

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

        # 第四行：耗时与预计耗时
        progress_grid.addWidget(QLabel("耗时:"), 3, 0)
        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        progress_grid.addWidget(self.elapsed_label, 3, 1)

        progress_grid.addWidget(QLabel("预计耗时:"), 3, 2)
        self.eta_label = QLabel("--:--:--")
        self.eta_label.setStyleSheet("color: #00aa00; font-weight: bold;")
        progress_grid.addWidget(self.eta_label, 3, 3)

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

        task = self._get_selected_task()

        # 检查data.yaml
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            errors.append("❌ 缺少 data.yaml")
            return {
                "valid": False,
                "errors": "\n".join(errors),
                "train_count": 0,
                "val_count": 0,
                "test_count": 0,
            }

        try:
            with open(data_yaml, "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            errors.append(f"❌ 解析 data.yaml 失败: {exc}")
            return {
                "valid": False,
                "errors": "\n".join(errors),
                "train_count": 0,
                "val_count": 0,
                "test_count": 0,
            }

        base_path_val = data_cfg.get("path", ".")
        data_dir = data_yaml.parent

        def _resolve_path(entry):
            path_obj = Path(entry)
            if path_obj.is_absolute():
                return path_obj
            return (data_dir / base_path_val / path_obj).resolve()

        def _count_images(folder: Path):
            if not folder.exists():
                return 0
            return len(
                [
                    p
                    for p in folder.rglob("*")
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
                ]
            )

        train_entry = data_cfg.get("train")
        val_entry = data_cfg.get("val")
        test_entry = data_cfg.get("test")

        if not train_entry or not val_entry:
            errors.append("❌ data.yaml 缺少 train/val 定义")
        else:
            train_path = _resolve_path(train_entry)
            val_path = _resolve_path(val_entry)

            if not train_path.exists():
                errors.append(f"❌ 训练集不存在: {train_path}")
            else:
                train_count = _count_images(train_path)
                if train_count == 0:
                    errors.append("⚠️  训练集为空")

            if not val_path.exists():
                errors.append(f"❌ 验证集不存在: {val_path}")
            else:
                val_count = _count_images(val_path)
                if val_count == 0:
                    errors.append("⚠️  验证集为空")

        if test_entry:
            test_path = _resolve_path(test_entry)
            if test_path.exists():
                test_count = _count_images(test_path)

        # 检查 labels 目录（检测/分割/姿态/OBB 常见结构，分类可忽略）
        if task in ["detect", "segment", "pose", "obb"]:
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
                    config = yaml.safe_load(f) or {}

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

                # 任务选择
                task_val = config.get("task", "detect")
                for idx, (_, value) in enumerate(self.task_options):
                    if value == task_val:
                        self.task_combo.setCurrentIndex(idx)
                        break
                self._update_task_combo_state()

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
                config = yaml.safe_load(f) or {}

            config.setdefault("model", {})
            config.setdefault("training", {})

            # 更新模型配置
            version_names = ["yolo11", "yolo9", "yolo8", "yolo12"]
            size_names = ["nano", "small", "medium", "large", "xlarge"]
            config["model"]["version"] = version_names[self.version_combo.currentIndex()]
            config["model"]["backbone"] = size_names[self.model_combo.currentIndex()]

            # 更新任务
            selected_task = self._get_selected_task()
            config["task"] = selected_task

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
        self.task_combo.currentIndexChanged.connect(self.on_task_changed)

    def _get_selected_task(self) -> str:
        return self.task_combo.currentData()

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
                    "所选目录中未找到 data.yaml 文件。\n\n"
                    "请按照 YOLO 官方格式提供 data.yaml，并在其中定义 path/train/val。",
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
        self.task_combo.setEnabled(False)

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
        self.elapsed_label.setText("00:00:00")
        self.eta_label.setText("--:--:--")

        # 重置耗时统计
        self.training_start_time = datetime.now()
        self.last_epoch_number = None
        self.last_epoch_timestamp = None
        self.epoch_time_accum = 0.0
        self.epoch_count = 0
        self.total_epochs_seen = None

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
                self.total_epochs_seen = total_epochs

                # 统计单epoch耗时
                now = datetime.now()
                if self.last_epoch_number is None:
                    self.last_epoch_number = epoch
                    self.last_epoch_timestamp = now
                elif epoch != self.last_epoch_number:
                    if self.last_epoch_timestamp:
                        delta_sec = (now - self.last_epoch_timestamp).total_seconds()
                        if delta_sec > 0:
                            self.epoch_time_accum += delta_sec
                            self.epoch_count += 1
                    self.last_epoch_number = epoch
                    self.last_epoch_timestamp = now

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

            # 更新耗时与预计耗时
            if self.training_start_time:
                elapsed_sec = (datetime.now() - self.training_start_time).total_seconds()
                self.elapsed_label.setText(self._format_duration(elapsed_sec))

                if self.total_epochs_seen and self.epoch_count > 0:
                    avg_epoch_sec = self.epoch_time_accum / self.epoch_count
                    eta_total_sec = avg_epoch_sec * self.total_epochs_seen
                    self.eta_label.setText(self._format_duration(eta_total_sec))
                else:
                    self.eta_label.setText("--:--:--")

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
                self.log("⚠️ 训练已停止。\n")

    def on_model_changed(self, index):
        """模型选择改变时的处理函数"""
        if not self.is_training:
            # 只在不训练时重新加载配置
            self.log(f"📋 模型已改变，加载对应的配置文件...")
            self._load_config()
            self._update_task_combo_state()

    def on_task_changed(self, index):
        if not self.is_training:
            selected = self._get_selected_task()
            self.log(f"📋 已选择任务: {selected}")

    def _update_task_combo_state(self):
        """根据 YOLO 版本限制任务选项（仅 YOLO11 支持分割/分类/姿势/OBB）"""
        allow_all = self.version_combo.currentIndex() == 0  # YOLO11
        model = self.task_combo.model()
        for i, (_, value) in enumerate(self.task_options):
            item = model.item(i)
            if item:
                item.setEnabled(allow_all or value == "detect")

        if not allow_all and self._get_selected_task() != "detect":
            self.task_combo.setCurrentIndex(0)
            self.log("⚠️  非 YOLO11 仅支持检测，已切换到检测任务")

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
        self.task_combo.setEnabled(True)

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

    def _format_duration(self, seconds: float) -> str:
        seconds = int(max(0, seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
