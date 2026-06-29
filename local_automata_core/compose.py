"""MCP 構築ヘルパ（フロントエンド中立・再利用ツールキットの一部）。

エージェント合成（compose_agent）・Agent ループ・ワークフローは利用側（local-automata）へ
移した。core にはどのフロントエンド/エージェントでも共通して必要な MCP の起動・取り込みだけを
残す。

**OS 層は local-aios（独立サービス）に委譲する**。core はアプリを自前で発見・起動せず、
`local-aios serve`（OS ゲートウェイ）を **単一の MCP サーバとして起動・接続**し、その配下に
現れるツール（syscall ＋ 各アプリのツール）を取り込む。発見・lazy 起動・スキーマキャッシュは
すべて OS 側（local-aios）が担う。
"""
from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

from .settings import AgentConfig

if TYPE_CHECKING:
    from .mcp_client import MCPManager


def _noop_log(event: str, detail: str) -> None:
    pass


def _gateway_server(dirs: list[str], explicit_servers: dict) -> dict:
    """`local-aios serve` を起動する単一 MCP サーバ定義（cfg）を作る。

    dirs（AIOS フォルダ）は serve の発見対象に、明示 MCP サーバ（agent.toml の
    `[mcp.servers.*]`）は mcpServers JSON に書き出して `--servers` で OS へ渡す。
    起動は現在の Python（`-m local_aios.cli`）で行う（local-aios は core の依存）。
    """
    args = ["-m", "local_aios.cli", "serve", *dirs]
    if explicit_servers:
        from .constants import project_cache_dir

        cache = project_cache_dir()
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, "aios_explicit_servers.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mcpServers": explicit_servers}, f, ensure_ascii=False)
        args += ["--servers", path]
    return {
        "command": sys.executable,
        "args": args,
        "description": "local-AIOS OS gateway (apps routed through local-aios serve)",
    }


def build_mcp(
    agent_config: AgentConfig, runtime: dict, *, log=_noop_log, workspace: str | None = None
) -> "tuple[MCPManager, list, MCPManager | None]":
    """OS ゲートウェイ（local-aios serve）へ接続し、(mcp, 提示ツール, None) を返す。

    旧 mcp_mode（auto/on_demand/eager）は廃止。OS が lazy 起動を担い、明示制御も
    OS の syscall ツール（list_apps/activate_app/deactivate_app）として提示されるため、
    core 側は「ゲートウェイへ接続して配下ツールを取り込む」一本に集約する。
    workspace: 出力先パラメータを持つツールに注入する作業ディレクトリ（ゲートウェイ越しでも
    引数はそのままアプリへ転送されるため、core 側 _wrap での注入が有効）。
    """
    from .mcp_client import MCPManager, resolve_call_timeout

    call_timeout = resolve_call_timeout(runtime.get("mcp_call_timeout", 120.0))
    mcp_dir = runtime.get("mcp_dir")
    dirs = [os.path.abspath(mcp_dir)] if mcp_dir else []
    explicit = agent_config.mcp_servers
    ws = os.path.abspath(workspace) if workspace else None

    if not explicit and not dirs:
        return MCPManager({}, call_timeout=call_timeout, workspace=ws), [], None

    gateway = _gateway_server(dirs, explicit)
    mcp = MCPManager({"localaios": gateway}, call_timeout=call_timeout, workspace=ws)
    mcp.start(log=log)  # ゲートウェイへ接続（OS が各アプリを lazy 起動）
    return mcp, mcp.tools(), None
