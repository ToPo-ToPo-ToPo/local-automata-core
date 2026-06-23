"""agent-core が使うローカル定数・ヘルパー。"""
from __future__ import annotations

import os


def project_cache_dir() -> str:
    """プロジェクト内（カレントディレクトリ）のキャッシュ/一時ディレクトリ `./.agent-core`。

    MCP スキーマキャッシュや一時ファイルをホーム（`~/.cache`）ではなくここに置き、
    自前の書き込みを起動ディレクトリの中で完結させる。ディレクトリは呼び出し側が必要時に作る。
    """
    return os.path.join(os.getcwd(), ".agent-core")
