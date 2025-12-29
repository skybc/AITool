"""
输出日志系统 - 将训练输出同时写到多个目标

支持：
1. StringIO缓冲区（用于消费者解析）
2. 日志文件（用于查看完整日志）
   - 按大小轮换：每个文件5MB
   - 按时间轮换：每天午夜轮换
   - 保留备份：最多100个备份文件
3. GUI信号（用于实时显示）
4. 标准输出（用于控制台显示）
"""

import sys
import io
import logging
from pathlib import Path
from typing import Optional, Callable
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class TeeWriter:
    """分流写入器 - 同时写到多个目标
    
    将输出写到：
    1. StringIO缓冲区（用于消费者解析）
    2. 日志文件（用于保存完整日志）
    3. 标准输出（用于控制台显示）
    """
    
    def __init__(self, 
                 string_buffer: io.StringIO,
                 console_output: bool = True):
        """初始化分流写入器
        
        Args:
            string_buffer: StringIO对象，用于消费者解析
            console_output: 是否输出到控制台
        """
        self.string_buffer = string_buffer
        self.console_output = console_output
        self.logger = logging.getLogger('train')
    
    def write(self, text: str) -> int:
        """写入文本到所有目标
        
        Args:
            text: 要写入的文本
            
        Returns:
            写入的字符数
        """
        if not text:
            return 0
        
        # 1. 写到StringIO缓冲区（用于消费者解析）
        self.string_buffer.write(text)
        
        # 2. 写到日志文件
        if text.strip():  # 只记录非空文本
            self.logger.info(text.rstrip())
        
        # 3. 写到控制台（可选）
        if self.console_output:
            sys.__stdout__.write(text)
        
        return len(text)
    
    def flush(self) -> None:
        """刷新所有缓冲区"""
        self.string_buffer.flush()
        if self.console_output:
            sys.__stdout__.flush()
    
    def isatty(self) -> bool:
        """判断是否为TTY"""
        return False


class OutputLogger:
    """输出日志管理器
    
    配置日志系统，为训练过程提供完整的日志记录
    
    日志轮换策略：
    - 按大小轮换：单个文件达到5MB时自动轮换
    - 按时间轮换：每天午夜自动轮换
    - 保留备份：最多保留100个备份文件（共101个）
    """
    
    # 日志配置常量
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    BACKUP_COUNT = 100  # 保留100个备份
    
    @staticmethod
    def setup_logger(log_dir: Path, log_name: str = 'train') -> logging.Logger:
        """设置日志系统
        
        Args:
            log_dir: 日志文件保存目录
            log_name: 日志器名称
            
        Returns:
            配置好的日志器
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        log_file = log_dir / f'{log_name}.log'
        
        # 使用RotatingFileHandler：按大小轮换
        # 当文件达到5MB时，自动轮换，保留100个备份
        size_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=OutputLogger.MAX_FILE_SIZE,
            backupCount=OutputLogger.BACKUP_COUNT,
            encoding='utf-8'
        )
        size_handler.setLevel(logging.DEBUG)
        
        # 使用TimedRotatingFileHandler：按时间轮换
        # 每天午夜(midnight)轮换，保留100个备份
        time_handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when='midnight',  # 每天午夜轮换
            interval=1,  # 间隔1天
            backupCount=OutputLogger.BACKUP_COUNT,
            encoding='utf-8'
        )
        time_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式：包含时间、级别、消息
        formatter = logging.Formatter(
            fmt='[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        size_handler.setFormatter(formatter)
        time_handler.setFormatter(formatter)
        
        # 添加两个处理器：同时实现大小轮换和时间轮换
        logger.addHandler(size_handler)
        logger.addHandler(time_handler)
        
        return logger
    
    @staticmethod
    def create_tee_writer(string_buffer: io.StringIO,
                         log_dir: Optional[Path] = None) -> TeeWriter:
        """创建分流写入器并配置日志
        
        Args:
            string_buffer: StringIO缓冲区
            log_dir: 日志目录（为None时不写日志文件）
            
        Returns:
            配置好的TeeWriter实例
        """
        # 如果指定了日志目录，则配置日志系统
        if log_dir:
            OutputLogger.setup_logger(log_dir, log_name='train')
        
        return TeeWriter(
            string_buffer=string_buffer,
            console_output=True
        )
