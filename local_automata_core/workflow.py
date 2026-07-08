"""オーケストレーション型ワークフローの**仕様モデルとパーサ**（再利用ツールキットの一部）。

外部 YAML で「決まった手順」を定義する WorkflowSpec / WorkflowStep と、その検証付きパーサ
parse_workflow を提供する。settings.load_agent_config が agent.toml の workflow_file を読む際に
使う。仕様に従って Agent を駆動する WorkflowRunner（Agent 結合）は利用側（フロントエンド）に
ある（core は Agent ループを持たない）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

WORKFLOW_MODES = ("advisory", "orchestrated")


@dataclass
class OnFail:
    """ステップが result='fail' を返したときの遷移先とリトライ上限。"""

    goto: str
    max_retries: int = 3


@dataclass
class WorkflowStep:
    id: str
    title: str
    instructions: str
    allowed_tools: list[str] | None = None  # None なら全ツール
    max_steps: int | None = None            # None ならエージェント既定
    on_fail: OnFail | None = None


@dataclass
class WorkflowSpec:
    name: str
    description: str = ""
    system_append: str | None = None
    steps: list[WorkflowStep] = field(default_factory=list)


# --- パース ---------------------------------------------------------------


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _parse_step(raw: Any, index: int) -> WorkflowStep:
    if not isinstance(raw, dict):
        raise ValueError(f"steps[{index}] must be a map")
    step_id = _require_str(raw.get("id"), f"steps[{index}].id")
    instructions = _require_str(raw.get("instructions"), f"steps[{index}].instructions")
    title = raw.get("title") or step_id
    if not isinstance(title, str):
        raise ValueError(f"steps[{index}].title must be a string")

    allowed = raw.get("allowed_tools")
    if allowed is not None and not (
        isinstance(allowed, list) and all(isinstance(t, str) for t in allowed)
    ):
        raise ValueError(f"steps[{index}].allowed_tools must be a list of strings")

    max_steps = raw.get("max_steps")
    if max_steps is not None and not (isinstance(max_steps, int) and max_steps > 0):
        raise ValueError(f"steps[{index}].max_steps must be an integer of 1 or greater")

    on_fail = None
    raw_fail = raw.get("on_fail")
    if raw_fail is not None:
        if not isinstance(raw_fail, dict) or "goto" not in raw_fail:
            raise ValueError(f"steps[{index}].on_fail must be a map containing goto")
        retries = raw_fail.get("max_retries", 3)
        if not (isinstance(retries, int) and retries >= 0):
            raise ValueError(
                f"steps[{index}].on_fail.max_retries must be an integer of 0 or greater"
            )
        on_fail = OnFail(goto=_require_str(raw_fail["goto"], f"steps[{index}].on_fail.goto"),
                         max_retries=retries)

    return WorkflowStep(
        id=step_id,
        title=title,
        instructions=instructions,
        allowed_tools=allowed,
        max_steps=max_steps,
        on_fail=on_fail,
    )


def parse_workflow(text: str, *, label: str = "workflow") -> WorkflowSpec:
    """YAML 文字列を WorkflowSpec に変換し、内容を検証する。"""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse {label} YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be written as a map (key: value format)")

    mode = data.get("mode", "orchestrated")
    if mode not in WORKFLOW_MODES:
        raise ValueError(f"mode must be one of {list(WORKFLOW_MODES)}")
    if mode != "orchestrated":
        raise ValueError(
            "YAML workflows only support mode: orchestrated"
            " (for advisory mode, use a Markdown workflow_file)"
        )

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("steps must be a list containing at least one step")
    steps = [_parse_step(s, i) for i, s in enumerate(raw_steps)]

    ids = [s.id for s in steps]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"Duplicate step id in steps: {sorted(dup)}")
    valid = set(ids)
    for s in steps:
        if s.on_fail and s.on_fail.goto not in valid:
            raise ValueError(
                f"steps[{s.id}].on_fail.goto refers to an unknown id: {s.on_fail.goto}"
            )

    system_append = data.get("system_append")
    if system_append is not None and not isinstance(system_append, str):
        raise ValueError("system_append must be a string")

    return WorkflowSpec(
        name=str(data.get("name") or "workflow"),
        description=str(data.get("description") or ""),
        system_append=system_append,
        steps=steps,
    )
