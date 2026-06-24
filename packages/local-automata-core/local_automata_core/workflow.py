"""オーケストレーション型ワークフロー。

外部 YAML で「決まった手順」を定義し、各ステップを順番に・決定論的に実行する。
助言型（system プロンプトへ手順を注入し LLM の自律判断に委ねる方式: settings の
workflow_file = "*.md"）と異なり、こちらはステップの順序・遷移・終了をコード側が
握るため、指示したワークフローを確実に実行させたい場合に使う。

ステップの完了は complete_step 制御ツール（tools/workflow_control.py）で報告させ、
呼ばれるまで次へ進めない。result='fail' のステップは on_fail.goto へ戻す。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from .agent import Agent
from .images import build_user_content
from .tools.workflow_control import COMPLETE_STEP_TOOL

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


# --- 実行 -----------------------------------------------------------------


class WorkflowRunner:
    """WorkflowSpec に従って Agent をステップごとに駆動する。

    Agent と同じ run()/reset()/on_text/llm を備え、CLI・Web から透過的に使える。
    complete_step が呼ばれるまで次のステップへ進まないため、手順を確実に踏ませる。
    """

    def __init__(self, agent: Agent, spec: WorkflowSpec, base_system: str) -> None:
        self.agent = agent
        self.spec = spec
        self._base_system = base_system

    # CLI / Web から Agent と同じインターフェイスで扱えるよう委譲する。
    @property
    def llm(self):  # noqa: D401 - 委譲プロパティ
        return self.agent.llm

    @property
    def on_text(self):
        return self.agent.on_text

    @on_text.setter
    def on_text(self, value) -> None:
        self.agent.on_text = value

    @property
    def on_status(self):
        return self.agent.on_status

    @on_status.setter
    def on_status(self, value) -> None:
        self.agent.on_status = value

    def reset(self) -> None:
        self.agent.set_system(self._base_system)
        self.agent.reset()

    def _compose_system(self, step: WorkflowStep, index: int) -> str:
        parts = [self._base_system]
        if self.spec.system_append:
            parts.append(self.spec.system_append)
        n = len(self.spec.steps)
        parts.append(
            f"## 現在のステップ ({index + 1}/{n}): {step.title}\n"
            f"{step.instructions}\n\n"
            "このステップに集中し、完了したら complete_step(result=\"pass\") を必ず呼んで"
            "ください。やり直しや前段の修正が必要なら complete_step(result=\"fail\", "
            "note=理由) を呼んでください。"
        )
        return "\n\n".join(parts)

    def run(
        self,
        user_input: str,
        images: list[str] | None = None,
        files: list[str] | None = None,
    ) -> str:
        index_of = {s.id: i for i, s in enumerate(self.spec.steps)}
        retries: dict[str, int] = {}
        idx = 0
        first = True
        final = ""
        emit = self.agent.on_text
        while 0 <= idx < len(self.spec.steps):
            step = self.spec.steps[idx]
            emit(f"\n■ Step {idx + 1}/{len(self.spec.steps)}: {step.title}\n")
            if first:
                user_content: Any = build_user_content(user_input, images, files)
                first = False
            else:
                user_content = (
                    f"前のステップが完了しました。次のステップ「{step.title}」に進みます。"
                )
            allowed: set[str] | None = None
            if step.allowed_tools is not None:
                allowed = set(step.allowed_tools) | {COMPLETE_STEP_TOOL}
            final, control = self.agent.run_phase(
                user_content,
                system=self._compose_system(step, idx),
                allowed_tools=allowed,
                max_steps=step.max_steps or self.agent.config.max_steps,
                stop_tools={COMPLETE_STEP_TOOL},
            )
            if control is None:
                msg = (
                    f"(Workflow aborted: step \"{step.title}\" reached the limit "
                    "without calling complete_step)"
                )
                emit(msg + "\n")
                return msg
            if control.get("result") == "fail" and step.on_fail is not None:
                n = retries.get(step.id, 0)
                if n >= step.on_fail.max_retries:
                    msg = (
                        f"(Workflow aborted: step \"{step.title}\" reached the retry "
                        f"limit ({step.on_fail.max_retries} times))"
                    )
                    emit(msg + "\n")
                    return msg
                retries[step.id] = n + 1
                emit(
                    f"  ↻ result=fail → going back to "
                    f"\"{self.spec.steps[index_of[step.on_fail.goto]].title}\" "
                    f"({n + 1}/{step.on_fail.max_retries})\n"
                )
                idx = index_of[step.on_fail.goto]
                continue
            idx += 1
        emit("\n■ Workflow completed\n")
        return final
