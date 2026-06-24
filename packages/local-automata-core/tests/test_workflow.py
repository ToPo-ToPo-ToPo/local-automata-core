import textwrap

import pytest

from local_automata_core import Agent
from local_automata_core.config import Config
from local_automata_core.settings import load_agent_config
from local_automata_core.tools import build_registry
from local_automata_core.tools.workflow_control import build_workflow_control_tools
from local_automata_core.workflow import WorkflowRunner, parse_workflow
from conftest import FakeLLM, msg, tool_call

YAML = textwrap.dedent(
    """
    name: demo
    description: デモ
    mode: orchestrated
    system_append: 追加プロンプト
    steps:
      - id: implement
        title: 実装
        instructions: コードを書く
        allowed_tools: [read_file, write_file]
      - id: evaluate
        title: 評価
        instructions: 評価する
        on_fail:
          goto: implement
          max_retries: 1
    """
)


# --- パース ---------------------------------------------------------------


def test_parse_valid():
    spec = parse_workflow(YAML)
    assert spec.name == "demo"
    assert spec.system_append == "追加プロンプト"
    assert [s.id for s in spec.steps] == ["implement", "evaluate"]
    assert spec.steps[0].allowed_tools == ["read_file", "write_file"]
    assert spec.steps[1].on_fail.goto == "implement"
    assert spec.steps[1].on_fail.max_retries == 1
    # title 省略時は id を使う / max_steps は任意
    assert spec.steps[0].title == "実装"


def test_parse_rejects_advisory_mode():
    with pytest.raises(ValueError, match="orchestrated"):
        parse_workflow("mode: advisory\nsteps: []\n")


def test_parse_requires_steps():
    with pytest.raises(ValueError, match="steps"):
        parse_workflow("name: x\nmode: orchestrated\n")


def test_parse_rejects_duplicate_id():
    bad = textwrap.dedent(
        """
        name: x
        steps:
          - {id: a, instructions: i}
          - {id: a, instructions: j}
        """
    )
    with pytest.raises(ValueError, match="Duplicate"):
        parse_workflow(bad)


def test_parse_rejects_unknown_goto():
    bad = textwrap.dedent(
        """
        name: x
        steps:
          - {id: a, instructions: i, on_fail: {goto: zzz}}
        """
    )
    with pytest.raises(ValueError, match="unknown id"):
        parse_workflow(bad)


# --- 実行（WorkflowRunner） ----------------------------------------------


def make_runner(tmp_path, spec, responses):
    reg = build_registry(
        ["read_file", "write_file", "list_dir", "edit_file", "run_command"],
        str(tmp_path / "ws"),
    )
    for tool in build_workflow_control_tools():
        reg.register(tool)
    out = []
    agent = Agent(FakeLLM(responses), reg, Config(), on_text=out.append, system_prompt="BASE")
    runner = WorkflowRunner(agent, spec, "BASE")
    return runner, agent, out


def test_steps_run_in_order(tmp_path):
    spec = parse_workflow(YAML)
    responses = [
        msg(content="実装した", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
        msg(content="評価OK", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
    ]
    runner, agent, out = make_runner(tmp_path, spec, responses)
    final = runner.run("やって")
    assert final == "評価OK"
    joined = "".join(out)
    assert "Step 1/2: 実装" in joined
    assert "Step 2/2: 評価" in joined
    assert "Workflow completed" in joined


def test_phase_system_prompt_is_swapped(tmp_path):
    spec = parse_workflow(YAML)
    responses = [
        msg(content="a", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
        msg(content="b", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
    ]
    runner, agent, out = make_runner(tmp_path, spec, responses)
    runner.run("やって")
    # 最初の LLM 呼び出しで渡った system に BASE・追加プロンプト・ステップ指示が含まれる
    first_system = agent.llm.calls[0][0]["content"]
    assert "BASE" in first_system
    assert "追加プロンプト" in first_system
    assert "コードを書く" in first_system


def test_on_fail_loops_then_aborts(tmp_path):
    spec = parse_workflow(YAML)
    responses = [
        msg(content="impl1", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
        msg(content="eval1", tool_calls=[tool_call("complete_step", {"result": "fail"})]),
        msg(content="impl2", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
        msg(content="eval2", tool_calls=[tool_call("complete_step", {"result": "fail"})]),
    ]
    runner, agent, out = make_runner(tmp_path, spec, responses)
    final = runner.run("やって")
    assert "retry limit" in final
    assert "↻" in "".join(out)  # 1回戻った


def test_on_fail_recovers_and_completes(tmp_path):
    spec = parse_workflow(YAML)
    responses = [
        msg(content="impl1", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
        msg(content="eval1", tool_calls=[tool_call("complete_step", {"result": "fail"})]),
        msg(content="impl2", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
        msg(content="eval2", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
    ]
    runner, agent, out = make_runner(tmp_path, spec, responses)
    final = runner.run("やって")
    assert final == "eval2"
    assert "Workflow completed" in "".join(out)


def test_disallowed_tool_is_blocked(tmp_path):
    spec = parse_workflow(YAML)
    responses = [
        # evaluate ステップ（allowed_tools 未指定=全許可）はスキップさせず、
        # まず implement（read_file/write_file のみ許可）で run_command を試す
        msg(content="", tool_calls=[tool_call("run_command", {"command": "ls"})]),
        msg(content="やり直す", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
        msg(content="ok", tool_calls=[tool_call("complete_step", {"result": "pass"})]),
    ]
    runner, agent, out = make_runner(tmp_path, spec, responses)
    runner.run("やって")
    assert "cannot be used in this step" in "".join(out)


def test_max_steps_without_complete_aborts(tmp_path):
    spec = parse_workflow(
        "name: x\nsteps:\n  - {id: a, instructions: i, max_steps: 2}\n"
    )
    responses = [
        msg(content="考え中1"),
        msg(content="考え中2"),
    ]
    runner, agent, out = make_runner(tmp_path, spec, responses)
    final = runner.run("やって")
    assert "reached the limit" in final


# --- settings 連携 --------------------------------------------------------


def _write(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def test_settings_loads_yaml_as_spec(tmp_path):
    _write(tmp_path / "workflow.yaml", YAML)
    _write(tmp_path / "agent.toml", 'workflow_file = "workflow.yaml"\n')
    cfg = load_agent_config(tmp_path / "agent.toml")
    assert cfg.workflow is None
    assert cfg.workflow_spec is not None
    assert cfg.workflow_spec.name == "demo"


def test_settings_loads_markdown_as_advisory(tmp_path):
    _write(tmp_path / "workflow.md", "# 手順\n1. やる\n")
    _write(tmp_path / "agent.toml", 'workflow_file = "workflow.md"\n')
    cfg = load_agent_config(tmp_path / "agent.toml")
    assert cfg.workflow_spec is None
    assert cfg.workflow is not None and "手順" in cfg.workflow
