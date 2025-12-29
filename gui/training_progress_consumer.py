"""
训练进度消费者 - 专门负责解析和处理训练进度数据

从输出中提取：
- Epoch 进度
- GPU 内存占用
- 损失值（box_loss, cls_loss, dfl_loss）
- 检测实例数
- 输入图像尺寸

处理ANSI转义码和各种输出格式
"""

import re
from core.output_consumer import OutputConsumer


class TrainingProgressConsumer(OutputConsumer):
    """训练进度消费者
    
    解析 YOLO 的训练输出，提取实时训练指标
    
    处理以下格式：
    - 带ANSI转义码：\x1b[K   1/6   0G   2.325   10.21   1.62   1106   640
    - 回车分隔符：\r
    - 多行进度数据
    """
    
    # ANSI转义码模式
    ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*[mKHJ]')
    
    def __init__(self):
        """初始化消费者"""
        self.progress_data = {
            'epoch': None,
            'total_epochs': None,
            'gpu_mem': None,
            'box_loss': None,
            'cls_loss': None,
            'dfl_loss': None,
            'instances': None,
            'size': None,
        }
        
        # 用于缓存上一次的数据，避免重复更新
        self._last_line = None
    
    def consume(self, output: str) -> None:
        """消费并解析输出
        
        Args:
            output: 新增的输出内容
        
        YOLO 输出格式示例：
        1/6         0G      2.412      10.14      1.451       1568        640: 10% ━─────────── 10/97 8.9s/it
        
        原始数据可能包含ANSI转义码和回车符：
        \x1b[K        1/6         0G      2.325      10.21       1.62       1106        640: 0%\r
        
        格式：Epoch  GPUMem  BoxLoss  ClsLoss  DFLLoss  Instances  Size
        """
        if not output:
            return
        
        # 清除ANSI转义码
        cleaned_output = self._remove_ansi_codes(output)
        
        # 使用 \r 或 \n 分割行
        # 由于进度条使用\r覆盖上一行，我们要找最后一条有效的进度行
        lines = re.split(r'[\r\n]+', cleaned_output)
        
        # 倒序遍历，获取最新的数据
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            
            # 避免重复解析同一行
            if line == self._last_line:
                continue
            
            self._last_line = line
            
            # 尝试解析进度行
            if self._parse_progress_line(line):
                break
    
    def _remove_ansi_codes(self, text: str) -> str:
        """移除ANSI转义码
        
        Args:
            text: 包含ANSI码的文本
            
        Returns:
            清理后的文本
        """
        return self.ANSI_PATTERN.sub('', text)
    
    def _parse_progress_line(self, line: str) -> bool:
        """解析完整的进度行
        
        YOLO 标准输出格式（固定列）：
        Epoch  GPUMem  BoxLoss  ClsLoss  DFLLoss  Instances  Size: Progress
        1/6         0G      2.412      10.14      1.451       1568        640: 10%
        
        注意：行前可能有多个空格或其他字符
        
        Args:
            line: 清理后的输出行
            
        Returns:
            是否成功解析
        """
        # 匹配完整的进度行
        # 格式: "Epoch GPU Loss Loss Loss Instances Size: ..."
        # 行前可能有空白字符
        pattern = r'(\d+)/(\d+)\s+(\d+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)'
        match = re.search(pattern, line)
        
        if not match:
            return False
        
        try:
            self.progress_data['epoch'] = int(match.group(1))
            self.progress_data['total_epochs'] = int(match.group(2))
            self.progress_data['gpu_mem'] = float(match.group(3))
            self.progress_data['box_loss'] = float(match.group(4))
            self.progress_data['cls_loss'] = float(match.group(5))
            self.progress_data['dfl_loss'] = float(match.group(6))
            self.progress_data['instances'] = float(match.group(7))
            self.progress_data['size'] = int(match.group(8))
            return True
        except (ValueError, IndexError):
            return False
    
    def get_progress_data(self):
        """获取当前解析的进度数据
        
        Returns:
            字典，包含所有解析的指标
        """
        return self.progress_data.copy()
    
    def get_value(self, key: str, default=None):
        """获取特定指标的值
        
        Args:
            key: 指标名称（epoch, gpu_mem, box_loss, etc.）
            default: 如果指标为 None 时的默认值
        
        Returns:
            指标值或默认值
        """
        value = self.progress_data.get(key)
        return value if value is not None else default
    
    def on_training_start(self) -> None:
        """训练开始时重置数据"""
        self.progress_data = {
            'epoch': None,
            'total_epochs': None,
            'gpu_mem': None,
            'box_loss': None,
            'cls_loss': None,
            'dfl_loss': None,
            'instances': None,
            'size': None,
        }
        self._last_line = None
    
    def on_training_end(self) -> None:
        """训练结束时的处理"""
        pass
