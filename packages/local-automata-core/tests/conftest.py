"""テスト用の共通フェイク。LLM をネットワークなしで差し替える。"""
from __future__ import annotations

import json
from types import SimpleNamespace


def tool_call(name: str, args: dict, call_id: str = "c1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def msg(content=None, tool_calls=None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class FakeLLM:
    """chat(messages, tools, on_text) を順番に返答するフェイク LLM。

    responses: 各ターンで返す SimpleNamespace のリスト（msg(...) で作る）。
    on_text には content を一括で渡す（表示確認用）。
    """

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools, on_text=lambda t: None):
        self.calls.append(list(messages))
        m = self._responses.pop(0)
        if m.content:
            on_text(m.content)
        return m
