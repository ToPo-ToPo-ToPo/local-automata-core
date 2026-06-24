"""agent-core — エージェント共有コア（L3）。

ツールフレームワーク・共通ツール・MCP 連携・エージェントループ・文脈管理を提供する。
LLM への接続（生成・tool-calling プロトコル）は L2 の local-llm-client に委ね、推論サーバー
（ゲートウェイ）は別パッケージ local-llm-server。

    from local_automata_core import Agent, Config, build_registry, compose_agent
    from local_automata_core import load_agent_config

複数フロントエンド（CLI / Web 等）から再利用できる中立なコア。LLM 本文は on_text、
ループの進捗は on_status で受け取り、フロントエンドが表示を差し替える。
"""
from __future__ import annotations

from local_llm_client import LLMClient

from .agent import SYSTEM_PROMPT, Agent
from .compose import build_mcp, compose_agent
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
    attach_agentic_tools,
    build_memory_tools,
    build_plan_tools,
    build_registry,
    default_registry,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "SYSTEM_PROMPT",
    "Config",
    "LLMClient",
    "compose_agent",
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
    "attach_agentic_tools",
    "build_memory_tools",
    "MemoryStore",
    "build_plan_tools",
]
