"""local-automata-core — エージェント共有ツールキット（L3）。

汎用ツール（filesystem / shell / memory / plan）・MCP 連携・LLMClient・設定・STT・画像/PDF 変換・
ワークフロー仕様パーサ・文脈ユーティリティを提供する**再利用ツールキット**。**Agent ループ本体は
持たない**——ループ・合成（compose_agent）・WorkflowRunner・エージェント結合ツールは、それらを使う
側（フロントエンド local-automata や各エージェント）が所有する。LLM 接続は L2 local-llm-client、
推論サーバーは別パッケージ local-llm-server。

    from local_automata_core import Config, LLMClient, build_registry, build_mcp
    from local_automata_core import load_agent_config, to_image_url

複数のエージェント/フロントエンドが、これらの部品の上に自分のループを組む。
"""
from __future__ import annotations

from local_llm_client import LLMClient

from .compose import build_mcp
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

__version__ = "0.6.0"

__all__ = [
    "Config",
    "LLMClient",
    "build_mcp",
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
