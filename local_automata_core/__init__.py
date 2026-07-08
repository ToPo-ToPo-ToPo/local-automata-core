"""local-automata-core — エージェント構築用の共有ツールキット。

汎用ツール・MCP 連携・設定・STT・画像/PDF 変換・ワークフロー仕様パーサを提供する。
**Agent ループ本体は持たない** — ループ・合成・WorkflowRunner は利用側（フロントエンド）が所有する。

    from local_automata_core import Config, LLMClient, build_registry
    from local_automata_core import load_agent_config, to_image_url
"""
from __future__ import annotations

from local_llm_client import LLMClient

from .config import Config
from .images import (
    build_user_content,
    pdf_to_image_urls,
    to_image_url,
    video_to_image_urls,
)
from .settings import AgentConfig, find_config_path, load_agent_config
from .stt import DEFAULT_STT_MODEL, Transcriber
from .tools import (
    MemoryStore,
    Tool,
    ToolRegistry,
    build_memory_tools,
    build_plan_tools,
    build_registry,
    default_registry,
)

try:
    from importlib.metadata import version

    __version__ = version("local-automata-core")
except Exception:  # インストールされていない作業ツリーからの実行
    __version__ = "0.0.0+unknown"

__all__ = [
    "Config",
    "LLMClient",
    "AgentConfig",
    "load_agent_config",
    "find_config_path",
    "Transcriber",
    "DEFAULT_STT_MODEL",
    "build_user_content",
    "to_image_url",
    "video_to_image_urls",
    "pdf_to_image_urls",
    "Tool",
    "ToolRegistry",
    "build_registry",
    "default_registry",
    "build_memory_tools",
    "MemoryStore",
    "build_plan_tools",
]
