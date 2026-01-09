"""CLI key bindings module.

实现自定义快捷键行为:
- Ctrl+C x2: 双击退出
- Esc: 停止生成（有任务时）或清空输入（无任务时）
- Esc Esc (快速连按): 第一次停止生成，第二次清空输入
"""

import time
from prompt_toolkit.key_binding import KeyBindings


class CLIKeyBindings:
    """CLI 自定义快捷键绑定管理器."""

    # 双击检测的时间窗口（秒）
    DOUBLE_CLICK_TIMEOUT = 0.5

    def __init__(self, cli_app):
        """初始化快捷键绑定.

        Args:
            cli_app: CLIApplication 实例的引用
        """
        self._cli_app = cli_app
        self._bindings = KeyBindings()
        self._last_ctrl_c_time = 0
        self._setup_bindings()

    def _setup_bindings(self):
        """注册自定义快捷键."""

        # Esc: 停止生成或清空输入
        @self._bindings.add("escape")
        def _(event):
            """停止生成（有任务）或清空输入（无任务）."""
            app = self._cli_app
            if app.current_task and not app.current_task.done():
                app.current_task.cancel()
                app.status.notify_warning("Stopped (Esc again to clear)")
            else:
                buffer = event.app.current_buffer
                buffer.cursor_position = 0
                buffer.delete(count=len(buffer.document.current_line))

        # Esc Esc: 强制清空输入（需要快速连按两次）
        @self._bindings.add("escape", "escape")
        def _(event):
            """强制清空输入行."""
            buffer = event.app.current_buffer
            buffer.cursor_position = 0
            buffer.delete(count=len(buffer.document.current_line))

    @property
    def bindings(self) -> KeyBindings:
        """获取 KeyBindings 对象."""
        return self._bindings

    def handle_ctrl_c(self) -> bool:
        """处理 Ctrl+C，返回是否应该退出.

        Returns:
            True 表示应该退出程序，False 表示只是取消当前任务
        """
        current_time = time.time()
        time_since_last = current_time - self._last_ctrl_c_time
        self._last_ctrl_c_time = current_time

        # 如果在时间窗口内再次按 Ctrl+C，则退出
        if time_since_last < self.DOUBLE_CLICK_TIMEOUT:
            return True
        return False
