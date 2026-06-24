"""フロントエンド中立なエージェント合成（composition root）。

`Agent` はコア（フロントエンド非依存。コールバックは注入可能）。本モジュールはその上の
「フル装備のエージェントを組む配線」——ツールレジストリ構築・memory/plan/MCP/workflow
ツールの登録・LLMClient 生成・Agent 構築——を、**特定のフロントエンドに依存しない形**で提供する。

CLI（cli.py）・Web GUI（webapp.py）・外部組み込み（別アプリ）は、いずれもこの
`compose_agent(...)` を呼び、自分の提示先（端末 / SSE / その他）に応じた
`on_text` / `on_status` / `log` を渡す。既定はいずれも no-op（中立）で、CLI 由来の
stdout/stderr ストリーマは持ち込まない（それは cli.py 側の既定）。

三層:
  - コア:   Agent / Config / LLMClient / ToolRegistry（フロントエンド非依存）
  - 合成:   compose_agent（本モジュール・中立）
  - 提示:   cli._stream/_log、webapp のリクエスト別 sink、組み込み側の SSE 等
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from local_llm_client import LLMClient

from .agent import Agent
from .config import Config
from .settings import AgentConfig
from .tools import (
    MemoryStore,
    Workspace,
    build_memory_tools,
    build_plan_tools,
    build_registry,
)
from .tools.workflow_control import build_workflow_control_tools
from .workflow import WorkflowRunner

if TYPE_CHECKING:
    from .mcp_client import MCPManager


def _noop_log(event: str, detail: str) -> None:
    pass


def _noop_text(text: str) -> None:
    pass


def _noop_status(status: str) -> None:
    pass


def compose_agent(
    config: Config,
    agent_config: AgentConfig,
    workdir: str | Workspace,
    planning: bool,
    extra_tools: list | None = None,
    mcp: "MCPManager | None" = None,
    on_text=_noop_text,
    on_status=_noop_status,
    log=_noop_log,
) -> Agent | WorkflowRunner:
    """フル装備のエージェントを組み立てる（フロントエンド中立）。

    提示用コールバック（on_text / on_status / log）は呼び出し側が渡す。既定は no-op。
    workflow_spec（YAML, orchestrated）が指定されていれば WorkflowRunner を、それ以外は
    通常の Agent（advisory ワークフローは system へ注入）を返す。どちらも run()/reset()/
    on_text/llm を備える。

    extra_tools に渡したツールはそのまま登録される（カスタムツールの登録口）。
    """
    # サブエージェント探索が同じ Workspace を共有できるよう、ここで確定させる。
    ws = workdir if isinstance(workdir, Workspace) else Workspace(workdir)
    tools = build_registry(agent_config.tools, ws, config.allow_install)
    if agent_config.memory and agent_config.memory.enabled:
        store = MemoryStore(agent_config.memory.path)
        for tool in build_memory_tools(store):
            tools.register(tool)
    if planning:
        for tool in build_plan_tools():
            tools.register(tool)
    for tool in extra_tools or []:
        tools.register(tool)
    # 方式B: MCP を遅延起動で渡したとき、探索／起動のメタツールを登録する。
    # 実ツールは activate_mcp_server 呼び出し時に動的に registry へ加わる。
    if mcp is not None:
        from .mcp_client import build_mcp_meta_tools

        for tool in build_mcp_meta_tools(mcp, tools, log):
            tools.register(tool)
    orchestrated = agent_config.workflow_spec is not None
    if orchestrated:
        for tool in build_workflow_control_tools():
            tools.register(tool)
    kwargs: dict = {"planning": planning, "workflow": agent_config.workflow}
    if agent_config.system_prompt is not None:
        kwargs["system_prompt"] = agent_config.system_prompt
    # 接続専用クライアント（local-llm-client）を Config から組み立てる。
    # 読み取りは長め/無制限（ローカルの巨大モデルは初回応答が遅い）、接続は短く。
    read = config.request_timeout
    timeout = httpx.Timeout(read if read and read > 0 else None, connect=10.0)
    llm = LLMClient(
        config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=timeout,
        tool_mode=config.tool_mode,
        enable_thinking=config.enable_thinking,
        stream=config.stream,
    )
    agent = Agent(llm, tools, config, log=log, on_text=on_text, on_status=on_status, **kwargs)
    # 大規模コード調査用のサブエージェント探索／生成画像の視認ツールを後付けで追加。
    # llm/ws/親agent を参照するため build_registry ではなくここで登録する。
    if tools.get("read_file") is not None:
        from .tools import attach_agentic_tools

        attach_agentic_tools(agent, ws)
    if orchestrated:
        return WorkflowRunner(agent, agent_config.workflow_spec, agent._base_system)
    return agent


def _mcp_cache_path() -> str:
    """auto モードのツールスキーマキャッシュの保存先（プロジェクト内 `./.agent-core/`）。"""
    import os
    from .constants import project_cache_dir
    return os.path.join(project_cache_dir(), "mcp_schema_cache.json")


def build_mcp(
    agent_config: AgentConfig, runtime: dict, *, log=_noop_log
) -> "tuple[MCPManager, list, MCPManager | None]":
    """MCPManager を構築・起動し、(mcp, compose_agent に渡す extra_tools, メタツール用 mcp) を返す。

    mcp_mode: auto（既定）=全ツールを最初から提示しプロセスは初回呼び出し時に起動 /
    on_demand=エージェントが list/activate で明示起動 / eager=起動時に全接続。
    表示は log コールバックへ（フロントエンドが差し替え。既定は無音）。
    """
    import os
    from .mcp_client import MCPManager, resolve_call_timeout

    call_timeout = resolve_call_timeout(runtime.get("mcp_call_timeout", 120.0))
    mcp_dir = runtime.get("mcp_dir")
    dirs = [os.path.abspath(mcp_dir)] if mcp_dir else []
    mode = runtime.get("mcp_mode", "auto")
    mcp = MCPManager(
        agent_config.mcp_servers,
        call_timeout=call_timeout,
        dirs=dirs,
        cache_path=_mcp_cache_path(),
    )
    if not agent_config.mcp_servers and not dirs:
        return mcp, [], None
    if mode == "eager":
        mcp.start(log=log)
        return mcp, mcp.tools(), None
    if mode == "on_demand":
        mcp.start_lazy(log=log)
        return mcp, [], mcp
    mcp.start_lazy(log=log)
    return mcp, mcp.lazy_tools(log=log), None
