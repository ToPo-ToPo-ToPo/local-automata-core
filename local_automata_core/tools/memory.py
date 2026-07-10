from __future__ import annotations

import json
from pathlib import Path

from .base import Tool

# メモリ機能のツール名（設定で有効化されたときのみ登録される）
MEMORY_TOOL_NAMES = ("remember", "recall")


class MemoryStore:
    """JSON ファイルに永続化する、追記型の単純なメモリ。

    セッションをまたいで「覚えておくべきこと」を保存・参照する。
    プロファイルごとに別ファイルを指定すれば、キャラクター単位の記憶になる。
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def _load(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    def _save(self, items: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def remember(self, content: str) -> str:
        # 重複ガード: モデルは同じ内容で remember を繰り返し呼ぶことがある
        # (エージェントループはツールの反復呼び出しを制限しないため、1ターンで
        # 同一内容が複数回保存される事故が実際に起きた)。空白差だけの一致も弾き、
        # 「既に記憶済み」を明示的に返してモデルに反復をやめる合図を出す。
        items = self._load()
        normalized = content.strip()
        if any(m.strip() == normalized for m in items):
            return f"既に同じ内容を記憶済みです（現在 {len(items)} 件）。"
        items.append(content)
        self._save(items)
        return f"記憶しました（現在 {len(items)} 件）。"

    def recall(self, query: str | None = None) -> str:
        items = self._load()
        if query:
            needle = query.lower()
            items = [m for m in items if needle in m.lower()]
        if not items:
            return "（該当する記憶はありません）"
        return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(items))


def build_memory_tools(store: MemoryStore) -> list[Tool]:
    """メモリストアに束ねた remember / recall ツールを返す。"""
    return [
        Tool(
            name="remember",
            description="後で思い出せるように事実や好み・経緯などを長期メモリに保存する。",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "記憶する内容"},
                },
                "required": ["content"],
            },
            func=store.remember,
        ),
        Tool(
            name="recall",
            description="長期メモリを参照する。会話の冒頭や関連する話題で使うとよい。query で絞り込める。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "絞り込みキーワード（省略時は全件）"},
                },
            },
            func=store.recall,
        ),
    ]
