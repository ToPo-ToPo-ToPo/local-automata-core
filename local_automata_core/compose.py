"""MCP 構築ヘルパ（フロントエンド中立・再利用ツールキットの一部）。

エージェント合成（compose_agent）・Agent ループ・ワークフローは利用側（local-automata）へ
移した。core にはどのフロントエンド/エージェントでも共通して必要な MCP の起動・取り込みだけを
残す（設定 → MCPManager ＋ 提示するツール群）。
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .settings import AgentConfig

if TYPE_CHECKING:
    from .mcp_client import MCPManager


def _noop_log(event: str, detail: str) -> None:
    pass


def _mcp_cache_path() -> str:
    """auto モードのツールスキーマキャッシュの保存先（プロジェクト内キャッシュ）。"""
    from .constants import project_cache_dir

    return os.path.join(project_cache_dir(), "mcp_schema_cache.json")


def build_mcp(
    agent_config: AgentConfig, runtime: dict, *, log=_noop_log
) -> "tuple[MCPManager, list, MCPManager | None]":
    """MCPManager を構築・起動し、(mcp, 提示する extra_tools, メタツール用 mcp) を返す。

    mcp_mode: auto（既定）=全ツールを最初から提示しプロセスは初回呼び出し時に起動 /
    on_demand=エージェントが list/activate で明示起動 / eager=起動時に全接続。
    表示は log コールバックへ（フロントエンドが差し替え。既定は無音）。
    """
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
