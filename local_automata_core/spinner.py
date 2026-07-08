from __future__ import annotations

import itertools
import sys
import threading
import time
from typing import IO

# 端末スピナーを使うか（グローバル）。GUI では本文が SSE で
# ブラウザへ流れるため、端末へスピナーを描くと改行・行管理だけが残って空白が溜まる。
# その場合は set_enabled(False) で全スピナーを無効化する。CLI は既定で有効。
_ENABLED = True


def set_enabled(value: bool) -> None:
    """以降に生成するスピナーの有効/無効をまとめて切り替える。"""
    global _ENABLED
    _ENABLED = bool(value)


class Spinner:
    """生成中・実行中を示す簡易スピナー（stderr、TTY のときのみ動作）。

    idle_delay 秒のあいだ出力が無いとスピナーを表示し、出力が来たら消す。
      - idle_delay=0  : すぐ表示（ツール実行中など、出力が来ない区間向け）
      - idle_delay>0  : 出力が途切れたときだけ表示（生成のストール検知向け）

    ストリーミング中は emit のたびに notify_before()/notify_after() を呼ぶ。
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(
        self, label: str = "", stream: IO[str] | None = None, idle_delay: float = 0.0
    ) -> None:
        self._label = label
        self._stream = stream or sys.stderr
        self._enabled = _ENABLED and bool(getattr(self._stream, "isatty", lambda: False)())
        self._idle_delay = idle_delay
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last = time.monotonic()
        self._started_at = time.monotonic()
        self._visible = False
        self._at_line_start = True  # カーソルが行頭（新しい行）にあるか

    def start(self) -> None:
        if not self._enabled or self._thread is not None:
            return
        self._stop.clear()
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        frames = itertools.cycle(self.FRAMES)
        while not self._stop.wait(0.1):
            with self._lock:
                if time.monotonic() - self._last < self._idle_delay:
                    continue
                if not self._visible and not self._at_line_start:
                    # 直前の出力が改行で終わっていない場合は行を分けてから描く
                    self._stream.write("\n")
                    self._at_line_start = True
                elapsed = int(time.monotonic() - self._started_at)
                self._stream.write(f"\r{next(frames)} {self._label} ({elapsed}s)")
                self._stream.flush()
                self._visible = True

    def notify_before(self) -> None:
        """実出力を書く直前に呼ぶ。表示中のスピナーを消す。"""
        if not self._enabled:
            return
        with self._lock:
            if self._visible:
                self._stream.write("\r\033[K")
                self._stream.flush()
                self._visible = False

    def notify_after(self, ended_with_newline: bool) -> None:
        """実出力を書いた直後に呼ぶ。アイドル計測と行状態を更新する。"""
        if not self._enabled:
            return
        with self._lock:
            self._last = time.monotonic()
            self._at_line_start = ended_with_newline

    def stop(self) -> None:
        if not self._enabled or self._stop.is_set():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            if self._visible:
                self._stream.write("\r\033[K")
                self._stream.flush()
                self._visible = False

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
