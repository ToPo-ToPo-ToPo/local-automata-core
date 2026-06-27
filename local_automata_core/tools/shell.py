from __future__ import annotations

import os
import re
import signal
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
        # shell=True の子プロセス（/bin/sh）は複合コマンド（`a && b`、パイプ等）だと
        # 自身を exec せず孫プロセスを fork する。subprocess.run の timeout は直下の sh しか
        # kill しないため、孫（例: 無限ループの python）が orphan として CPU を食い続ける。
        # これを防ぐため、新しいセッション（プロセスグループ）で起動し、タイムアウト時は
        # グループ全体に SIGKILL を送って子孫ごと確実に始末する。
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(ws.root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_command_env(),
            start_new_session=True,  # 子をプロセスグループのリーダーにする（getpgid==pid）
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # 直下の sh だけでなく、その配下の孫プロセスも含めてグループごと kill する。
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            # ゾンビを残さないよう刈り取る（パイプの後始末も兼ねる）。
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return f"Error: command timed out after {timeout}s"
        # 出力長の上限・切り詰めは中央（Agent._execute）で全ツール一律に行う。ここでは
        # 生の出力をそのまま返す（中央が先頭＋末尾を残して中央を省略する）。
        output = (stdout or "") + (stderr or "")
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
