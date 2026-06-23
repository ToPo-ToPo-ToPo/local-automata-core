from __future__ import annotations

from typing import Any

from .base import Tool

PLAN_TOOL_NAMES = ("update_plan",)

_MARKS = {"done": "✓", "in_progress": "▶", "pending": "□"}


def _render(steps: list[dict[str, str]]) -> str:
    lines = [f"{_MARKS.get(s['status'], '□')} {s['step']}" for s in steps]
    return "\n".join(lines) or "(計画が空です)"


def _update_plan(steps: Any) -> str:
    if not isinstance(steps, list):
        return "Error: steps はリストで指定してください"
    normalized: list[dict[str, str]] = []
    for item in steps:
        if isinstance(item, str):
            normalized.append({"step": item, "status": "pending"})
        elif isinstance(item, dict):
            # step / task / description / content / title のいずれかを本文として受ける
            text = (
                item.get("step")
                or item.get("task")
                or item.get("description")
                or item.get("content")
                or item.get("title")
            )
            if text is None:
                return "Error: 各ステップは文字列か {step(または task), status} で指定してください"
            status = item.get("status", "pending")
            if status not in _MARKS:
                status = "pending"
            normalized.append({"step": str(text), "status": status})
        else:
            return "Error: 各ステップは文字列か {step(または task), status} で指定してください"
    return "計画:\n" + _render(normalized)


def build_plan_tools() -> list[Tool]:
    """着手前の計画（チェックリスト）を作成・更新する update_plan ツールを返す。"""
    return [
        Tool(
            name="update_plan",
            description=(
                "タスクを小さなステップに分解した計画（チェックリスト）を作成・更新する。"
                "着手前に作り、進捗に応じて status を更新する。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "ステップの一覧",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string", "description": "ステップの内容"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "done"],
                                    "description": "状態",
                                },
                            },
                            "required": ["step", "status"],
                        },
                    },
                },
                "required": ["steps"],
            },
            func=_update_plan,
        ),
    ]
