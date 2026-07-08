# local-automata-core

エージェント構築用の共有ツールキット（import 名 `local_automata_core`）。汎用ツール（filesystem / shell / memory / plan）・MCP 連携・設定ローダ・STT・画像/PDF 変換・ワークフロー仕様パーサを提供する。**Agent ループ本体は含まない** — ループや合成は利用側が組む。

## 関連パッケージ

本パッケージはツール・MCP 連携・設定・文脈管理を提供し、次と組み合わせて使う。

| パッケージ | 役割 |
|---|---|
| [local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-llm-client) | LLM 接続クライアント |

推論サーバーは別パッケージ [local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server)。

## インストール

```bash
uv add local-automata-core
```

## 使い方

```python
from local_automata_core import find_config_path, load_agent_config, default_registry

config = load_agent_config(find_config_path())  # agent.toml + AGENTS.md を読む
registry = default_registry()                   # 汎用ツールを workspace/ に構築
```

利用側はこれらの部品の上に独自の Agent ループを組む。

## 開発

```bash
uv sync
uv run pytest
```

## ライセンス

Apache-2.0
