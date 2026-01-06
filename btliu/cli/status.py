"""CLI 状态管理模块."""

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Optional

from prompt_toolkit.formatted_text import FormattedText


class StatusLevel(Enum):
    """状态级别."""

    IDLE = "idle"
    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    PROGRESS = "progress"


@dataclass
class StatusState:
    """状态快照."""

    level: StatusLevel = StatusLevel.IDLE
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    details: Optional[str] = None


class CLIStatusManager:
    """CLI 状态管理器.

    特性:
    - 线程安全的状态存储
    - 支持 bottom_toolbar 动态获取状态
    - 简单的 API 用于状态更新
    """

    def __init__(self):
        self._state = StatusState()
        self._lock = Lock()

    def set_status(
        self, level: StatusLevel, message: str, details: Optional[str] = None
    ):
        """设置状态."""
        with self._lock:
            self._state = StatusState(
                level=level, message=message, timestamp=time.time(), details=details
            )

    def clear_status(self):
        """清除状态（回到空闲）."""
        self.set_status(StatusLevel.IDLE, "")

    def get_state(self) -> StatusState:
        """获取当前状态快照（线程安全）."""
        with self._lock:
            return StatusState(
                level=self._state.level,
                message=self._state.message,
                timestamp=self._state.timestamp,
                details=self._state.details,
            )

    def get_toolbar_text(self) -> FormattedText:
        """获取工具栏文本（用于 prompt 的 bottom_toolbar）.

        Returns:
            FormattedText with status message. Returns empty if idle.
        """
        state = self.get_state()

        if state.level == StatusLevel.IDLE or not state.message:
            return FormattedText([])

        # 根据 level 选择颜色（prompt_toolkit 样式）
        color_map = {
            StatusLevel.SUCCESS: "fg:#00ff00",  # 绿色
            StatusLevel.ERROR: "fg:#ff0000",  # 红色
            StatusLevel.WARNING: "fg:#ffff00",  # 黄色
            StatusLevel.PROGRESS: "fg:#ffaa00",  # 橙色
            StatusLevel.INFO: "",  # 终端原生
        }
        color = color_map.get(state.level, "")

        details_part = f" - {state.details}" if state.details else ""

        return FormattedText(
            [
                ("", "│ "),
                (color, state.message + details_part),
            ]
        )

    def notify_success(self, message: str, details: Optional[str] = None):
        self.set_status(StatusLevel.SUCCESS, message, details)

    def notify_info(self, message: str, details: Optional[str] = None):
        self.set_status(StatusLevel.INFO, message, details)

    def notify_warning(self, message: str, details: Optional[str] = None):
        self.set_status(StatusLevel.WARNING, message, details)

    def notify_error(self, message: str, details: Optional[str] = None):
        self.set_status(StatusLevel.ERROR, message, details)

    def notify_progress(self, message: str, details: Optional[str] = None):
        self.set_status(StatusLevel.PROGRESS, message, details)
