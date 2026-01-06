"""CLI 显示处理模块.

使用 prompt_toolkit 的 FormattedText 和 Style 实现优雅的格式化输出。
"""

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

# 全局样式定义
CLI_STYLE = Style.from_dict(
    {
        "success": "fg:#00ff00 bold",
        "error": "fg:#ff4444 bold",
        "info": "fg:#00ffff",
        "warning": "fg:#ffff00",
        "dim": "fg:#666666",
        "banner": "fg:#00ccff",
        "command": "fg:#ffaa00",
    }
)


class CLIDisplay:
    """CLI 显示处理器 - 负责所有输出.

    使用 prompt_toolkit 的 FormattedText 格式化，提供更优雅的输出体验。
    """

    def __init__(self, style: Style = CLI_STYLE):
        self._style = style

    def print_success(self, message: str, newline: bool = True):
        """打印成功消息."""
        text = FormattedText(
            [
                ("class:success", f"✓ {message}"),
                ("", "\n" if newline else ""),
            ]
        )
        print_formatted_text(text, style=self._style)

    def print_error(self, message: str, newline: bool = True):
        """打印错误消息."""
        text = FormattedText(
            [
                ("class:error", f"✗ {message}"),
                ("", "\n" if newline else ""),
            ]
        )
        print_formatted_text(text, style=self._style)

    def print_warning(self, message: str, newline: bool = True):
        """打印警告消息."""
        text = FormattedText(
            [
                ("class:warning", f"⚠ {message}"),
                ("", "\n" if newline else ""),
            ]
        )
        print_formatted_text(text, style=self._style)

    def print_info(self, message: str, newline: bool = True):
        """打印信息消息."""
        text = FormattedText(
            [
                ("class:info", message),
                ("", "\n" if newline else ""),
            ]
        )
        print_formatted_text(text, style=self._style)

    def print_banner(self, text: str):
        """打印启动横幅."""
        print_formatted_text(FormattedText([("class:banner", text)]), style=self._style)

    def print_commands(self, commands: list[str]):
        """打印命令列表."""
        parts = []
        for i, cmd in enumerate(commands):
            if i > 0:
                parts.append(("", " "))
            parts.append(("class:command", cmd))
        parts.append(("", "\n"))
        print_formatted_text(FormattedText(parts), style=self._style)

    def print_raw(self, text: str):
        """直接打印原始文本（用于不需要格式化的输出）."""
        print(text)

    def print_formatted(self, fragments: list[tuple[str, str]]):
        """打印自定义格式化文本.

        Args:
            fragments: [(style, text), ...] 列表
        """
        print_formatted_text(FormattedText(fragments), style=self._style)
