from __future__ import annotations

from ..settings import DEFAULT_WORKDIR
from .base import Tool, ToolRegistry
from .filesystem import FILESYSTEM_TOOL_NAMES, build_filesystem_tools
from .memory import MEMORY_TOOL_NAMES, MemoryStore, build_memory_tools
from .plan import PLAN_TOOL_NAMES, build_plan_tools
from .shell import SHELL_TOOL_NAMES, build_shell_tools
from .workspace import Workspace, session_timestamp, slugify

# workdir に束ねて使う組み込みツール名（メモリは [memory] 設定で別途有効化）
BUILTIN_TOOL_NAMES = (*FILESYSTEM_TOOL_NAMES, *SHELL_TOOL_NAMES)


def available_tool_names() -> list[str]:
    return list(BUILTIN_TOOL_NAMES)


def build_registry(
    names: list[str] | None = None,
    workdir: str | Workspace = DEFAULT_WORKDIR,
    allow_install: bool = False,
) -> ToolRegistry:
    """ツール名のリストから、workdir に閉じ込めた ToolRegistry を構築する。

    workdir には文字列パスのほか Workspace を渡せる（実行中に改名する用途）。
    names が None のときは全ツールを登録する。未知のツール名はエラー。
    1つでもツールを使う場合は workdir を作成する。
    allow_install=False のとき run_command はパッケージ導入コマンドをブロックする。
    """
    selected = available_tool_names() if names is None else names
    unknown = [name for name in selected if name not in BUILTIN_TOOL_NAMES]
    if unknown:
        if any(name in MEMORY_TOOL_NAMES for name in unknown):
            raise ValueError(
                f"メモリツール {list(MEMORY_TOOL_NAMES)} は tools に列挙せず、"
                "[memory] enabled=true で有効化してください。"
            )
        if any(name in PLAN_TOOL_NAMES for name in unknown):
            raise ValueError(
                f"計画ツール {list(PLAN_TOOL_NAMES)} は tools に列挙せず、"
                "planning=true で有効化してください。"
            )
        raise ValueError(
            f"unknown tool(s): {unknown}. available: {available_tool_names()}"
        )

    ws = workdir if isinstance(workdir, Workspace) else Workspace(workdir)
    if selected:
        ws.ensure()

    builders = {
        tool.name: tool
        for tool in (
            *build_filesystem_tools(ws),
            *build_shell_tools(ws, allow_install),
        )
    }
    registry = ToolRegistry()
    for name in selected:
        registry.register(builders[name])
    return registry


def default_registry(
    workdir: str = DEFAULT_WORKDIR, allow_install: bool = False
) -> ToolRegistry:
    return build_registry(None, workdir, allow_install)


# 注: エージェント結合ツール（dispatch_agent / view_image / read_pdf_pages）の後付け登録
# （attach_agentic_tools）は Agent ループとともに利用側（フロントエンド）へ移した。core は
# フロントエンド非依存の汎用ツール（build_registry が組む filesystem / shell など）だけを提供する。

__all__ = [
    "Tool",
    "ToolRegistry",
    "Workspace",
    "session_timestamp",
    "slugify",
    "BUILTIN_TOOL_NAMES",
    "available_tool_names",
    "build_registry",
    "default_registry",
    "MemoryStore",
    "build_memory_tools",
    "MEMORY_TOOL_NAMES",
    "build_plan_tools",
    "PLAN_TOOL_NAMES",
]
