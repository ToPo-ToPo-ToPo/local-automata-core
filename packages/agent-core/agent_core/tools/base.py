from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def truncate_middle(text: str, limit: int) -> str:
    """text を limit 文字以内に収める。先頭と末尾を残し、中央を省略マーカーに置換する。

    Codex の truncate_middle と同方針。エラーやトレースバックは出力の「末尾」に出る
    ことが多いため、先頭だけ残す素朴な切り詰め（text[:limit]）では肝心の情報が消える。
    末尾も保持することで、長い出力でも要点（先頭の文脈＋末尾のエラー）を失わない。

    残り予算は先頭6割・末尾4割で配分する。結果長はマーカー分だけ limit を上回りうる。
    """
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit * 3 // 5
    tail = limit - head
    omitted = len(text) - head - tail
    marker = f"\n…[中央 {omitted:,} 文字を省略 / 全 {len(text):,} 文字]…\n"
    if tail == 0:
        return text[:head] + marker
    return text[:head] + marker + text[-tail:]


@dataclass
class Tool:
    """エージェントが呼べる1つのツール。func は文字列（ツール結果）を返す。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    func: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def __iter__(self):
        return iter(self._tools.values())
