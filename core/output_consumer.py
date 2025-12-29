"""
输出消费者模式 - 用于处理训练过程中的标准输出捕获

提供一个发布-订阅架构：
- OutputConsumer: 消费者基类，定义消费者接口
- 多个消费者可以注册到训练线程
- 每个消费者独立解析输出数据
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class OutputConsumer(ABC):
    """输出消费者基类
    
    所有消费者必须继承此类并实现 consume() 方法
    """
    
    @abstractmethod
    def consume(self, output: str) -> None:
        """消费新的输出
        
        Args:
            output: 捕获的新输出内容（可能包含多行）
        
        注意：
        - output 是自上次 consume() 调用后新增的内容
        - 消费者需要自己维护状态和上下文
        - 实现应该足够高效，因为可能被频繁调用
        """
        pass
    
    def on_training_start(self) -> None:
        """训练开始时的回调（可选）"""
        pass
    
    def on_training_end(self) -> None:
        """训练结束时的回调（可选）"""
        pass


class OutputConsumerManager:
    """输出消费者管理器
    
    管理多个消费者的注册、注销和通知
    """
    
    def __init__(self):
        self._consumers: List[OutputConsumer] = []
        self._last_consumed_pos: int = 0
    
    def register(self, consumer: OutputConsumer) -> None:
        """注册一个消费者
        
        Args:
            consumer: OutputConsumer 实例
        """
        if consumer not in self._consumers:
            self._consumers.append(consumer)
            consumer.on_training_start()
    
    def unregister(self, consumer: OutputConsumer) -> None:
        """注销一个消费者
        
        Args:
            consumer: 要注销的 OutputConsumer 实例
        """
        if consumer in self._consumers:
            self._consumers.remove(consumer)
    
    def notify_all(self, full_output: str) -> None:
        """通知所有消费者新的输出
        
        Args:
            full_output: 完整的累积输出缓冲区内容
        
        只会将新增部分（自上次调用后）发送给消费者
        """
        # 计算新增部分
        new_content = full_output[self._last_consumed_pos:]
        
        if new_content:
            # 通知所有消费者
            for consumer in self._consumers:
                try:
                    consumer.consume(new_content)
                except Exception as e:
                    # 防止单个消费者的异常影响其他消费者
                    print(f"消费者异常 {consumer.__class__.__name__}: {e}")
        
        # 更新已消费位置
        self._last_consumed_pos = len(full_output)
    
    def notify_training_end(self) -> None:
        """通知所有消费者训练结束"""
        for consumer in self._consumers:
            try:
                consumer.on_training_end()
            except Exception as e:
                print(f"消费者 {consumer.__class__.__name__} 结束回调异常: {e}")
    
    def clear(self) -> None:
        """清空所有消费者"""
        self._consumers.clear()
        self._last_consumed_pos = 0
