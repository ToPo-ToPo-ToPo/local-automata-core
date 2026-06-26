# local-automata-core

エージェント共有コア（L3、import 名 `local_automata_core`）。ツール・MCP・エージェントループ・文脈管理。
LLM 接続は [local-llm-client](https://github.com/ToPo-ToPo-ToPo/local-llm-client)（L2、PyPI）、推論サーバーは別リポ
[local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server)。

単一パッケージのリポジトリ（配布名 `local-automata-core` / import 名 `local_automata_core`）。
利用側（フロントエンド [local-automata](https://github.com/ToPo-ToPo-ToPo/local-automata) 等）は PyPI から
`local-automata-core>=0.3.0` で依存する。

## 開発

```bash
uv sync          # 依存を同期（local-llm-client は PyPI から解決）
uv run pytest    # テスト
```
