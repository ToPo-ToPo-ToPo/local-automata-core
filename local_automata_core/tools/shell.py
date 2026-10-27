from __future__ import annotations

import os
import re
import signal
import shutil
import subprocess
import sys
from pathlib import Path

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


def find_git_bash() -> str | None:
    """Windows で Git for Windows の bash を探す（無ければ None。posix では常に None）。

    頭脳（LLM）が書くシェルコマンドは `mkdir -p`・`cp`・`&&` のような POSIX の書き方が既定で、
    cmd.exe では `-p` という名前のフォルダができる・`cp` が無いと崩れる。bash があればそれで動かし、
    macOS（/bin/sh）と同じ書き方を通す。

    PATH の ``bash`` は使わない: Windows では System32 の bash.exe（WSL の起動口）や WindowsApps の
    別名に当たり、ディストリビューションが無いと動かない。git の場所から Git の導入フォルダを辿り、
    その ``bin/bash.exe``（coreutils を PATH に足してから bash を起こす入口）を使う。
    """
    if os.name != "nt":
        return None
    roots: list[Path] = []
    git = shutil.which("git")
    if git:
        # <Git>/cmd/git.exe・<Git>/bin/git.exe・<Git>/mingw64/bin/git.exe のどれでも導入フォルダに届く
        roots.extend(list(Path(git).resolve().parents)[:3])
    for var in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        if os.environ.get(var):
            roots.append(Path(os.environ[var]) / "Git")
    if os.environ.get("LOCALAPPDATA"):
        roots.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Git")
    for root in roots:
        bash = root / "bin" / "bash.exe"
        if bash.is_file() and (root / "usr" / "bin").is_dir():
            return str(bash)
    return None


_SHELL_NOTES = {
    "ja": {
        "posix": "シェルは /bin/sh。",
        "bash": ("シェルは bash（Git for Windows）。mkdir -p・cp・ls・cat・&& などがそのまま使える。"
                 "パスは / 区切りで書く（例: C:/Users/...。円記号は bash がエスケープとして消す）。"),
        "cmd": ("シェルは Windows の cmd.exe（bash は無い）。mkdir -p・cp・ls・cat・rm は使えないので、"
                "mkdir（途中のフォルダも作る）・copy・dir・type・del を使う。"),
    },
    "en": {
        "posix": "The shell is /bin/sh.",
        "bash": ("The shell is bash (Git for Windows): mkdir -p, cp, ls, cat, && and the like work as usual. "
                 "Write paths with / (e.g. C:/Users/...); bash drops backslashes as escapes."),
        "cmd": ("The shell is Windows cmd.exe (no bash): mkdir -p, cp, ls, cat and rm do not exist, so use "
                "mkdir (creates intermediate folders), copy, dir, type and del."),
    },
}


def describe_shell(lang: str = "ja", bash: str | None = None) -> str:
    """run_command がどのシェルで動くかの一文（道具の説明に添えて頭脳に見せる）。

    bash を省くとこの機械で探す（build_shell_tools と同じ判定）。
    """
    notes = _SHELL_NOTES.get(lang, _SHELL_NOTES["en"])
    if os.name != "nt":
        return notes["posix"]
    return notes["bash"] if (bash or find_git_bash()) else notes["cmd"]


def _decode(data: bytes | None) -> str:
    """子の出力を文字列に。UTF-8 で読めなければ Windows のコンソールの文字コード（cmd.exe や
    Windows の道具のメッセージ。日本語版は cp932）で読み直す。以前は UTF-8 決め打ちで、
    cmd.exe の「'cp' は…認識されていません」が文字化けし、頭脳が失敗の理由を読めなかった。
    """
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if os.name == "nt":
        try:
            return data.decode("oem", errors="replace")
        except LookupError:
            pass
    return data.decode("utf-8", errors="replace")


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


def _popen_kwargs() -> dict:
    """子を「まとめて止められる単位」で起こす。

    posix: 新しいセッション（プロセスグループ）。タイムアウト時に killpg で孫まで届く。
    Windows: 新しいプロセスグループ（killpg 相当は無いので、停止は taskkill /T で木ごと）。
    """
    if os.name == "posix":
        return {"start_new_session": True}
    return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}


def _kill_tree(proc: subprocess.Popen) -> None:
    """直下の sh/cmd だけでなく、その配下の孫プロセスも含めて始末する。"""
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
            return
    # Windows: taskkill /T が子孫を辿って終わらせる（依存を増やさない）。失敗したら本人だけ kill
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        pass
    if proc.poll() is None:
        proc.kill()


def build_shell_tools(ws: Workspace, allow_install: bool = False) -> list[Tool]:
    """ws（workspace）を作業ディレクトリとして実行する run_command を返す。

    allow_install=False のときは、パッケージ導入とみなすコマンドをブロックする
    （勝手なインストールを防ぐ。--allow-install / CODER_ALLOW_INSTALL で解除）。
    注意: ヒューリスティックな検出であり、完全な隔離ではない（真の隔離はコンテナ/VM）。
    """
    bash = find_git_bash()  # Windows で Git の bash があれば POSIX の書き方で動かす（無ければ cmd.exe）

    def run_command(command: str, timeout: int = 120) -> str:
        if not allow_install and _looks_like_install(command):
            return (
                "[ブロック] パッケージのインストールは既定で無効です。"
                "許可する場合は --allow-install（または CODER_ALLOW_INSTALL=1）を付けて"
                "起動し直してください。標準ライブラリで代替できないか検討してください。\n"
                f"コマンド: {command}"
            )
        # シェルの子プロセス（/bin/sh・bash・cmd.exe）は複合コマンド（`a && b`、パイプ等）だと
        # 自身を exec せず孫プロセスを fork する。subprocess.run の timeout は直下のシェルしか
        # kill しないため、孫（例: 無限ループの python）が orphan として CPU を食い続ける。
        # これを防ぐため、まとめて止められる単位で起動し、タイムアウト時は子孫ごと確実に始末する。
        proc = subprocess.Popen(
            [bash, "-c", command] if bash else command,
            shell=not bash,
            cwd=str(ws.root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_command_env(),
            **_popen_kwargs(),
        )
        try:
            out_b, err_b = proc.communicate(timeout=timeout)
            stdout, stderr = _decode(out_b), _decode(err_b)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
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
            description=("workspace を作業ディレクトリとしてシェルコマンドを実行し、標準出力・標準エラー・終了コードを返す。"
                         + describe_shell("ja", bash)),
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


__all__ = ["SHELL_TOOL_NAMES", "build_shell_tools", "describe_shell", "find_git_bash"]
_ = sys
