# local-automata-core

エージェント構築用の共有ツールキット（import 名 `local_automata_core`）。汎用ツール（filesystem / shell / memory / plan）・MCP 連携・設定ローダ・画像/PDF 変換・ワークフロー仕様パーサを提供する。**Agent ループ本体は含まない** — ループや合成は利用側が組む。

LLM 接続は [local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-llm-client)、推論サーバーは [local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server) に委ねる。

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

## 開発

```bash
uv sync
uv run pytest
```

## ライセンス

Apache-2.0
