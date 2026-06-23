"""agent-core — エージェント共有コア（L3）。

ツールフレームワーク・共通ツール・MCP 連携・エージェントループ・文脈管理を提供する。
LLM への接続（生成・ツール呼び出しプロトコル）は L2 の local-llm-client に委ねる。

（移行中: local-automata の agent/ から段階的に抽出する）
"""
from __future__ import annotations

__version__ = "0.1.0"
