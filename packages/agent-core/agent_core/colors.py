from __future__ import annotations

import os
import sys
from typing import IO

RESET = "\033[0m"
BOLD_CYAN = "\033[1;36m"  # ツール呼び出し
DIM = "\033[2m"  # ツール結果・補助表示


def supports_color(stream: IO[str] | None = None) -> bool:
    """ANSI カラーを出してよいか判定する（出力先が TTY かつ NO_COLOR 未設定）。"""
    if os.environ.get("NO_COLOR"):
        return False
    stream = stream if stream is not None else sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def colorize(text: str, code: str, stream: IO[str] | None = None) -> str:
    """text を ANSI カラーで包む。色を出せない環境では素のまま返す。"""
    if not supports_color(stream):
        return text
    return f"{code}{text}{RESET}"
