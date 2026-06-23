from __future__ import annotations

import os
import re
import subprocess

from .base import Tool
from .workspace import Workspace


def _command_env() -> dict[str, str]:
    """run_command 用の環境変数。matplotlib を非対話バックエンドに固定する。

    生成コードが plt.show() を呼んでも GUI の別ウィンドウを開かせず、savefig で
    保存された画像だけを（Chat UI が拾って）表示させるため、MPLBACKEND=Agg を
    強制する。コードが明示的に matplotlib.use(...) しない限りこれが効く。
    """
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    return env

SHELL_TOOL_NAMES = ("run_command",)

# パッケージ導入とみなすコマンド（既定でブロックする）
_INSTALL_RE = re.compile(
    r"\b(?:"
    r"pip[0-9.]*\s+install"
    r"|python[0-9.]*\s+-m\s+pip\s+install"
    r"|uv\s+pip\s+install|uv\s+add|uv\s+sync"
    r"|conda\s+install|mamba\s+install|poetry\s+add"
    r"|npm\s+(?:install|i|add)|yarn\s+add|pnpm\s+(?:add|install)"
    r"|apt(?:-get)?\s+install|brew\s+install"
    r"|cargo\s+install|gem\s+install|go\s+install"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_install(command: str) -> bool:
    return bool(_INSTALL_RE.search(command))


def build_shell_tools(ws: Workspace, allow_install: bool = False) -> list[Tool]:
    """ws（workspace）を作業ディレクトリとして実行する run_command を返す。

    allow_install=False のときは、パッケージ導入とみなすコマンドをブロックする
    （勝手なインストールを防ぐ。--allow-install / CODER_ALLOW_INSTALL で解除）。
    注意: ヒューリスティックな検出であり、完全な隔離ではない（真の隔離はコンテナ/VM）。
    """

    def run_command(command: str, timeout: int = 120) -> str:
        if not allow_install and _looks_like_install(command):
            return (
                "[ブロック] パッケージのインストールは既定で無効です。"
                "許可する場合は --allow-install（または CODER_ALLOW_INSTALL=1）を付けて"
                "起動し直してください。標準ライブラリで代替できないか検討してください。\n"
                f"コマンド: {command}"
            )
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(ws.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_command_env(),
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        # 出力長の上限・切り詰めは中央（Agent._execute）で全ツール一律に行う。ここでは
        # 生の出力をそのまま返す（中央が先頭＋末尾を残して中央を省略する）。
        output = proc.stdout + proc.stderr
        if output.strip():
            return f"(exit code {proc.returncode})\n{output}"
        return f"(exit code {proc.returncode}, no output)"

    return [
        Tool(
            name="run_command",
            description="workspace を作業ディレクトリとしてシェルコマンドを実行し、標準出力・標準エラー・終了コードを返す。",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "実行するシェルコマンド"},
                    "timeout": {"type": "integer", "description": "タイムアウト秒数（既定120）"},
                },
                "required": ["command"],
            },
            func=run_command,
        ),
    ]
